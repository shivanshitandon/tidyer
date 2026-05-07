import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, PointStamped, Vector3Stamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from tf2_geometry_msgs import do_transform_point, do_transform_vector3
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass
class Detection2D3D:
    label: str
    shape: str
    centroid_uv: Tuple[int, int]
    area_px: float
    xyz_cam: Optional[Tuple[float, float, float]]
    yaw_rad: float


@dataclass
class BlockState:
    block_id: str
    label: str
    shape: str
    centroid_uv: Tuple[int, int]
    xyz_cam: Optional[Tuple[float, float, float]]
    area_px: float
    yaw_rad: float
    last_seen_s: float


class TidyerPerceptionNode(Node):
    """OpenCV-based perception pipeline.

    Captures triggered by services:
      /capture_reference  : snapshot the current frame as the target scene
      /capture_current    : snapshot, find the biggest moved block, publish
                            (pick, place) on /pick_place_pair as PoseArray
                            in base_link.
    """

    BASE_FRAME = 'base_link'

    def __init__(self) -> None:
        super().__init__('tidyer_perception')
        self.bridge = CvBridge()
        self.pub_pair = self.create_publisher(PoseArray, '/pick_place_pair', 10)

        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('camera_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('min_contour_area_px', 800.0)
        self.declare_parameter('position_tolerance_px', 50.0)
        self.declare_parameter('desk_plane_percentile', 50.0)
        self.declare_parameter('block_height_min_m', 0.005)
        self.declare_parameter('block_height_max_m', 0.15)
        self.declare_parameter('track_match_distance_px', 65.0)
        self.declare_parameter('state_output_path', '')

        # HSV config by label: [[h_lo,s_lo,v_lo],[h_hi,s_hi,v_hi]]
        self.declare_parameter(
            'hsv_ranges_json',
            json.dumps(
                {
                    # 0 93 93
                    # 28 103
                    # 'red': [[0, 100, 70], [12, 255, 255]],
                    'blue': [[90, 80, 50], [130, 255, 255]],
                    'green': [[60, 120, 60], [95, 255, 200]],
                    'yellow': [[18, 80, 80], [35, 255, 255]],
                }
            ),
        )

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_optical_frame = self.get_parameter('camera_optical_frame').value
        self.min_contour_area_px = float(self.get_parameter('min_contour_area_px').value)
        self.position_tolerance_px = float(self.get_parameter('position_tolerance_px').value)
        self.desk_plane_percentile = float(self.get_parameter('desk_plane_percentile').value)
        self.block_height_min_m = float(self.get_parameter('block_height_min_m').value)
        self.block_height_max_m = float(self.get_parameter('block_height_max_m').value)
        self.track_match_distance_px = float(self.get_parameter('track_match_distance_px').value)
        self.state_output_path = str(self.get_parameter('state_output_path').value)
        self.hsv_ranges: Dict[str, Tuple[np.ndarray, np.ndarray]] = self._load_hsv_ranges()

        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.reference_detections: List[Detection2D3D] = []
        self.reference_image: Optional[np.ndarray] = None
        self.block_states: Dict[str, BlockState] = {}
        self.next_track_id: int = 1

        self.pair_dir = Path.home() / 'final_proj' / 'pair'
        self.pair_dir.mkdir(parents=True, exist_ok=True)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, 10)
        self.create_subscription(Image, self.rgb_topic, self._rgb_cb, 10)
        self.create_subscription(Image, self.depth_topic, self._depth_cb, 10)

        self.create_service(Trigger, '/capture_reference', self._on_capture_reference)
        self.create_service(Trigger, '/capture_current', self._on_capture_current)

        # Background tracker for state snapshot/debug; does not publish goals.
        self.create_timer(0.5, self._tracker_tick)

        self.get_logger().info(
            'Tidyer perception ready. Trigger /capture_reference then /capture_current.'
        )

    def _load_hsv_ranges(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        raw = self.get_parameter('hsv_ranges_json').value
        parsed = json.loads(raw)
        ranges: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for label, bounds in parsed.items():
            lo, hi = bounds
            ranges[label] = (np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        return ranges

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def _rgb_cb(self, msg: Image) -> None:
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _depth_cb(self, msg: Image) -> None:
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _tracker_tick(self) -> None:
        if self.latest_rgb is None:
            return
        detections = self._segment_objects(self.latest_rgb, self.latest_depth)
        self._update_block_states(detections)
        self._write_state_snapshot()

    def _on_capture_reference(self, request, response):
        if self.latest_rgb is None:
            response.success = False
            response.message = 'No RGB frame yet.'
            return response
        rgb_snap = self.latest_rgb.copy()
        depth_snap = self.latest_depth
        self.reference_image = rgb_snap
        self.reference_detections = self._segment_objects(rgb_snap, depth_snap)
        response.success = True
        response.message = f'Captured {len(self.reference_detections)} reference objects.'
        self.get_logger().info(response.message)
        return response

    def _on_capture_current(self, request, response):
        if self.latest_rgb is None:
            response.success = False
            response.message = 'No RGB frame yet.'
            return response
        if not self.reference_detections:
            response.success = False
            response.message = 'No reference captured. Call /capture_reference first.'
            return response

        rgb_snap = self.latest_rgb.copy()
        depth_snap = self.latest_depth
        ts = time.strftime('%Y%m%d_%H%M%S')

        current = self._segment_objects(rgb_snap, depth_snap)
        moved = self._find_moved_objects(current, self.reference_detections)
        if not moved:
            response.success = True
            response.message = 'Scene aligned with reference (no moved blocks).'
            self.get_logger().info(response.message)
            return response

        # Biggest moved block first.
        moved.sort(key=lambda d: d[0].area_px, reverse=True)
        pick = moved[0][0]
        place = moved[0][1]  # paired reference slot for the moved block
        if pick.xyz_cam is None:
            response.success = False
            response.message = 'Missing depth for pick point.'
            return response
        if place.xyz_cam is None:
            response.success = False
            response.message = 'Missing depth for place point.'
            return response

        try:
            pick_base = self._camera_point_to_base(pick.xyz_cam)
            place_base = self._camera_point_to_base(place.xyz_cam)
        except TransformException as exc:
            response.success = False
            response.message = f'TF failed: {exc}'
            return response

        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.BASE_FRAME
        msg.poses.append(self._make_pose(pick_base, pick.yaw_rad))
        msg.poses.append(self._make_pose(place_base, place.yaw_rad))
        self.pub_pair.publish(msg)

        pair_subdir = self.pair_dir / f'pair_{ts}'
        pair_subdir.mkdir(parents=True, exist_ok=True)

        pick_uv = pick.centroid_uv
        pick_vis = rgb_snap.copy()
        cv2.circle(pick_vis, pick_uv, 12, (0, 0, 255), 2)
        cv2.circle(pick_vis, pick_uv, 3, (0, 0, 255), -1)
        cv2.putText(pick_vis, 'PICK', (pick_uv[0] + 14, pick_uv[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(str(pair_subdir / 'curr_pick.png'), pick_vis)

        place_uv = place.centroid_uv
        place_vis = self.reference_image.copy()
        cv2.circle(place_vis, place_uv, 12, (0, 255, 0), 2)
        cv2.circle(place_vis, place_uv, 3, (0, 255, 0), -1)
        cv2.putText(place_vis, 'PLACE', (place_uv[0] + 14, place_uv[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(pair_subdir / 'ref_place.png'), place_vis)

        metadata = {
            'timestamp': ts,
            'color': pick.label,
            'shape': pick.shape,
            'pick': {
                'pixel_uv': [int(pick_uv[0]), int(pick_uv[1])],
                'xyz_base_m': [float(pick_base[0]), float(pick_base[1]), float(pick_base[2])],
                'yaw_rad': float(pick.yaw_rad),
            },
            'place': {
                'pixel_uv': [int(place_uv[0]), int(place_uv[1])],
                'xyz_base_m': [float(place_base[0]), float(place_base[1]), float(place_base[2])],
                'yaw_rad': float(place.yaw_rad),
            },
        }
        (pair_subdir / 'pair.json').write_text(json.dumps(metadata, indent=2))

        response.success = True
        response.message = (
            f'Published pick {pick.label}/{pick.shape} -> place '
            f'(pick={pick_base}, place={place_base}, yaw={pick.yaw_rad:.2f})'
        )
        self.get_logger().info(response.message)
        return response

    def _segment_objects(self, bgr: np.ndarray, depth: Optional[np.ndarray]) -> List[Detection2D3D]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        desk_mask = self._compute_desk_block_mask(depth) #check TODO
        bgr_vis = bgr.copy()
        detections: List[Detection2D3D] = []
        for label, (lo, hi) in self.hsv_ranges.items():
            mask = cv2.inRange(hsv, lo, hi)
            if desk_mask is not None:
                mask = cv2.bitwise_and(mask, desk_mask)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_contour_area_px:
                    continue
                cv2.drawContours(bgr_vis, [contour], -1, (0, 255, 255), 2)
                cv2.imshow('Segmentation', bgr_vis)
                cv2.waitKey(1)
                moments = cv2.moments(contour)
                if moments['m00'] == 0:
                    continue
                u = int(moments['m10'] / moments['m00'])
                v = int(moments['m01'] / moments['m00'])
                cv2.circle(bgr_vis, (u, v), 4, (0, 0, 255), -1)
                xyz = self._block_top_xyz_camera(contour, u, v, depth)
                shape = self._classify_shape(contour, area)
                cv2.putText(
                    bgr_vis,
                    f'{label}:{shape}',
                    (u + 6, v - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                yaw = self._grasp_yaw(contour, depth)
                detections.append(
                    Detection2D3D(
                        label=label,
                        shape=shape,
                        centroid_uv=(u, v),
                        area_px=area,
                        xyz_cam=xyz,
                        yaw_rad=yaw,
                    )
                )
        cv2.imshow('Detections', bgr_vis)
        cv2.waitKey(1)
        return detections

    def _compute_desk_block_mask(self, depth_img: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if depth_img is None:
            return None
        depth_m = depth_img.astype(np.float32)
        if depth_img.dtype == np.uint16:
            depth_m = depth_m / 1000.0

        valid = depth_m > 0.0
        sample = depth_m[valid]
        if sample.size < 200:
            return None
        desk_depth = float(np.percentile(sample, self.desk_plane_percentile))

        # Keep points slightly above the desk plane where blocks usually sit.
        block_band = np.logical_and(
            depth_m >= max(0.0, desk_depth - self.block_height_max_m),
            depth_m <= max(0.0, desk_depth - self.block_height_min_m),
        )
        mask = (block_band.astype(np.uint8)) * 255
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _classify_shape(self, contour: np.ndarray, area: float) -> str:
        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            return 'unknown'

        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        vertices = len(approx)
        if vertices == 3:
            return 'triangle'
        if vertices == 4:
            (_, _), (w, h), _ = cv2.minAreaRect(contour)
            ratio = (w / float(h)) if h > 0 else 1.0
            if 0.85 <= ratio <= 1.15:
                return 'square'
            return 'rectangle'
        if vertices > 4:
            circularity = 4.0 * np.pi * area / (peri * peri)
            if circularity > 0.75:
                return 'circle'
            return 'polygon'
        return 'unknown'

    def _grasp_yaw(self, contour: np.ndarray, depth_img: Optional[np.ndarray]) -> float:
        """3D-PCA grasp yaw about base_link Z, in radians.

        Back-projects the block-top pixels through depth, runs PCA in
        camera_color_optical_frame, transforms the long-axis direction into
        base_link, and returns the yaw aligned with the long axis. Returns
        0.0 on any failure or near-symmetric block.
        """
        if depth_img is None:
            return 0.0
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return 0.0
        if len(contour) < 5:
            return 0.0

        mask = np.zeros(depth_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)

        ys_px, xs_px = np.where(mask > 0)
        if ys_px.size < 30:
            return 0.0

        depth_vals = depth_img[ys_px, xs_px]
        valid = depth_vals > 0
        ys_px = ys_px[valid]
        xs_px = xs_px[valid]
        depth_vals = depth_vals[valid]
        if depth_vals.size < 30:
            return 0.0

        # Restrict to the block-top plateau so PCA isn't biased by sides / desk leaks.
        is_uint16 = depth_img.dtype == np.uint16
        z_top_raw = float(np.percentile(depth_vals, 20))
        band_raw = 5.0 if is_uint16 else 0.005
        keep = depth_vals.astype(np.float32) <= z_top_raw + band_raw
        ys_px = ys_px[keep]
        xs_px = xs_px[keep]
        depth_vals = depth_vals[keep]
        if depth_vals.size < 30:
            return 0.0

        zs_m = depth_vals.astype(np.float32)
        if is_uint16:
            zs_m /= 1000.0
        xs_m = (xs_px.astype(np.float32) - self.cx) * zs_m / self.fx
        ys_m = (ys_px.astype(np.float32) - self.cy) * zs_m / self.fy
        pts3 = np.column_stack([xs_m, ys_m, zs_m])
        pts3 -= pts3.mean(axis=0)

        cov = (pts3.T @ pts3) / max(len(pts3) - 1, 1)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return 0.0

        # Bail on near-symmetric blocks: long axis is ill-defined and yaw flickers.
        if eigvals[-1] / max(eigvals[-2], 1e-9) < 1.21:
            return 0.0
        long_axis_cam = eigvecs[:, -1]

        try:
            tf = self.tf_buffer.lookup_transform(
                self.BASE_FRAME, self.camera_optical_frame, rclpy.time.Time()
            )
        except TransformException:
            return 0.0

        v_cam = Vector3Stamped()
        v_cam.header.stamp = rclpy.time.Time().to_msg()
        v_cam.header.frame_id = self.camera_optical_frame
        v_cam.vector.x = float(long_axis_cam[0])
        v_cam.vector.y = float(long_axis_cam[1])
        v_cam.vector.z = float(long_axis_cam[2])
        v_base = do_transform_vector3(v_cam, tf)

        long_angle_base = float(np.arctan2(v_base.vector.y, v_base.vector.x))
        # Long axis is bidirectional (period π); wrap to [-π/2, π/2).
        return float((long_angle_base + np.pi / 2.0) % np.pi - np.pi / 2.0)

    def _block_top_xyz_camera(
        self, contour: np.ndarray, u: int, v: int, depth_img: Optional[np.ndarray]
    ) -> Optional[Tuple[float, float, float]]:
        if depth_img is None:
            print("No depth image, cannot compute 3D position.")
            return None
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None: 
            print("Camera intrinsics not set, cannot compute 3D position.")
            return None
        if v < 0 or u < 0 or v >= depth_img.shape[0] or u >= depth_img.shape[1]:
            print("Invalid pixel coordinates, cannot compute 3D position.")
            return None

        # Sample depth inside the contour to avoid pulling in desk pixels.
        mask = np.zeros(depth_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        # Erode to stay clear of edges.
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1) #if shadow, increase this value
        depth_vals = depth_img[mask > 0]
        depth_vals = depth_vals[depth_vals > 0]
        if depth_vals.size == 0:
            print("No valid depth pixels in contour, cannot compute 3D position.")
            return None

        # Block top is the *closest* (smallest depth) plateau inside the contour.
        z_raw = float(np.percentile(depth_vals, 20))
        z_m = z_raw / 1000.0 if depth_img.dtype == np.uint16 else z_raw
        x_m = (u - self.cx) * z_m / self.fx
        y_m = (v - self.cy) * z_m / self.fy
        return (x_m, y_m, z_m)

    def _camera_point_to_base(self, xyz_cam: Tuple[float, float, float]) -> Tuple[float, float, float]:
        pt = PointStamped()
        pt.header.stamp = rclpy.time.Time().to_msg()  # latest available
        pt.header.frame_id = self.camera_optical_frame
        pt.point.x = float(xyz_cam[0])
        pt.point.y = float(xyz_cam[1])
        pt.point.z = float(xyz_cam[2])
        tf = self.tf_buffer.lookup_transform(
            self.BASE_FRAME, self.camera_optical_frame, rclpy.time.Time()
        )
        out = do_transform_point(pt, tf)
        return (out.point.x, out.point.y, out.point.z)

    def _make_pose(self, xyz_base: Tuple[float, float, float], yaw_rad: float) -> Pose:
        # Top-down EE: 180 deg about Y (lab5 convention; quat (0,1,0,0) at yaw=0),
        # then yaw about world Z.
        rot = R.from_euler('xyz', [0.0, np.pi, yaw_rad])
        qx, qy, qz, qw = rot.as_quat()
        pose = Pose()
        pose.position.x = float(xyz_base[0])
        pose.position.y = float(xyz_base[1])
        pose.position.z = float(xyz_base[2])
        pose.orientation.x = float(qx)
        pose.orientation.y = float(qy)
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)
        return pose

    # def _matching_reference(
    #     self, pick: Detection2D3D, reference: List[Detection2D3D]
    # ) -> Optional[Detection2D3D]:
    #     # Pair the moved block with the reference slot of same label/shape that
    #     # is FURTHEST from the pick centroid (i.e. the empty target slot).
    #     best = None
    #     best_dist = -1.0
    #     for ref in reference:
    #         if ref.label != pick.label or ref.shape != pick.shape:
    #             continue
    #         du = ref.centroid_uv[0] - pick.centroid_uv[0]
    #         dv = ref.centroid_uv[1] - pick.centroid_uv[1]
    #         dist = float(np.hypot(du, dv))
    #         if dist > best_dist:
    #             best_dist = dist
    #             best = ref
    #     return best

    def _find_moved_objects(
        self, current: List[Detection2D3D], reference: List[Detection2D3D]
    ) -> List[Tuple[Detection2D3D, Detection2D3D]]:
        moved: List[Tuple[Detection2D3D, Detection2D3D]] = []
        used_curr_idx = set()
        for ref in reference:
            best_idx = None
            best_dist = float('inf')
            for i, cur in enumerate(current):
                if i in used_curr_idx:
                    continue
                if cur.label != ref.label or cur.shape != ref.shape:
                    continue
                du = cur.centroid_uv[0] - ref.centroid_uv[0]
                dv = cur.centroid_uv[1] - ref.centroid_uv[1]
                dist = float(np.hypot(du, dv))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is None:
                continue
            used_curr_idx.add(best_idx)
            if best_dist > self.position_tolerance_px:
                moved.append((current[best_idx], ref))
        return moved

    def _update_block_states(self, detections: List[Detection2D3D]) -> None:
        now_s = self.get_clock().now().nanoseconds / 1e9
        unassigned = set(self.block_states.keys())
        updated: Dict[str, BlockState] = {}
        detections_sorted = sorted(detections, key=lambda d: d.area_px, reverse=True)

        for det in detections_sorted:
            best_id = None
            best_dist = float('inf')
            for block_id in list(unassigned):
                state = self.block_states[block_id]
                if state.label != det.label or state.shape != det.shape:
                    continue
                du = det.centroid_uv[0] - state.centroid_uv[0]
                dv = det.centroid_uv[1] - state.centroid_uv[1]
                dist = float(np.hypot(du, dv))
                if dist < best_dist:
                    best_dist = dist
                    best_id = block_id

            if best_id is not None and best_dist <= self.track_match_distance_px:
                prev = self.block_states[best_id]
                xyz = det.xyz_cam if det.xyz_cam is not None else prev.xyz_cam
                updated[best_id] = BlockState(
                    block_id=best_id,
                    label=det.label,
                    shape=det.shape,
                    centroid_uv=det.centroid_uv,
                    xyz_cam=xyz,
                    area_px=det.area_px,
                    yaw_rad=det.yaw_rad,
                    last_seen_s=now_s,
                )
                unassigned.remove(best_id)
                continue

            new_id = f'block_{self.next_track_id:03d}'
            self.next_track_id += 1
            updated[new_id] = BlockState(
                block_id=new_id,
                label=det.label,
                shape=det.shape,
                centroid_uv=det.centroid_uv,
                xyz_cam=det.xyz_cam,
                area_px=det.area_px,
                yaw_rad=det.yaw_rad,
                last_seen_s=now_s,
            )

        for block_id in unassigned:
            prev = self.block_states[block_id]
            if now_s - prev.last_seen_s <= 2.0:
                updated[block_id] = prev

        self.block_states = updated

    def _write_state_snapshot(self) -> None:
        if not self.state_output_path:
            return
        if not self.block_states:
            return

        blocks = []
        for block in self.block_states.values():
            blocks.append(
                {
                    'id': block.block_id,
                    'color': block.label,
                    'shape': block.shape,
                    'centroid_uv': [int(block.centroid_uv[0]), int(block.centroid_uv[1])],
                    'xyz_cam_m': list(block.xyz_cam) if block.xyz_cam is not None else None,
                    'area_px': float(block.area_px),
                    'yaw_rad': float(block.yaw_rad),
                    'last_seen_s': float(block.last_seen_s),
                }
            )

        relative = []
        block_list = list(self.block_states.values())
        for i in range(len(block_list)):
            for j in range(i + 1, len(block_list)):
                a = block_list[i]
                b = block_list[j]
                if a.xyz_cam is None or b.xyz_cam is None:
                    continue
                relative.append(
                    {
                        'from': a.block_id,
                        'to': b.block_id,
                        'delta_xyz_m': [
                            float(b.xyz_cam[0] - a.xyz_cam[0]),
                            float(b.xyz_cam[1] - a.xyz_cam[1]),
                            float(b.xyz_cam[2] - a.xyz_cam[2]),
                        ],
                    }
                )

        snapshot = {
            'timestamp_s': float(self.get_clock().now().nanoseconds / 1e9),
            'blocks': blocks,
            'relative_positions': relative,
        }

        out_path = Path(self.state_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(snapshot, indent=2))


def main(args=None):
    rclpy.init(args=args)
    node = TidyerPerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
