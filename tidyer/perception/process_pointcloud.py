import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class Detection2D3D:
    label: str
    centroid_uv: Tuple[int, int]
    bbox_xywh: Tuple[int, int, int, int]
    area_px: float
    xyz_cam: Optional[Tuple[float, float, float]]


class TidyerPerceptionNode(Node):
    """OpenCV-based perception pipeline for target/current scene differencing."""

    def __init__(self) -> None:
        super().__init__('tidyer_perception')
        self.bridge = CvBridge()
        self.pub_pose = self.create_publisher(PointStamped, '/cube_pose', 10)

        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('reference_image_path', '')
        self.declare_parameter('min_contour_area_px', 800.0)
        self.declare_parameter('position_tolerance_px', 50.0)

        # HSV config by label: [[h_lo,s_lo,v_lo],[h_hi,s_hi,v_hi]]
        self.declare_parameter(
            'hsv_ranges_json',
            json.dumps(
                {
                    'red': [[0, 100, 70], [12, 255, 255]],
                    'blue': [[90, 80, 50], [130, 255, 255]],
                    'green': [[35, 70, 50], [85, 255, 255]],
                    'yellow': [[18, 80, 80], [35, 255, 255]],
                }
            ),
        )

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.reference_image_path = self.get_parameter('reference_image_path').value
        self.min_contour_area_px = float(self.get_parameter('min_contour_area_px').value)
        self.position_tolerance_px = float(self.get_parameter('position_tolerance_px').value)
        self.hsv_ranges: Dict[str, Tuple[np.ndarray, np.ndarray]] = self._load_hsv_ranges()

        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.reference_bgr: Optional[np.ndarray] = self._load_reference_image()
        self.reference_detections: List[Detection2D3D] = []

        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, 10)
        self.create_subscription(Image, self.rgb_topic, self._rgb_cb, 10)
        self.create_subscription(Image, self.depth_topic, self._depth_cb, 10)
        self.create_timer(0.5, self._process_tick)

        self.get_logger().info('Tidyer perception ready (OpenCV segmentation; no VLM/YOLO).')

    def _load_hsv_ranges(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        raw = self.get_parameter('hsv_ranges_json').value
        parsed = json.loads(raw)
        ranges: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for label, bounds in parsed.items():
            lo, hi = bounds
            ranges[label] = (np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        return ranges

    def _load_reference_image(self) -> Optional[np.ndarray]:
        if not self.reference_image_path:
            self.get_logger().warn('No reference image configured. Set reference_image_path parameter.')
            return None
        ref_path = Path(self.reference_image_path)
        if not ref_path.exists():
            self.get_logger().warn(f'Reference image not found: {ref_path}')
            return None
        img = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().warn(f'Failed to decode reference image: {ref_path}')
            return None
        self.get_logger().info(f'Loaded reference image: {ref_path}')
        return img

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def _rgb_cb(self, msg: Image) -> None:
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _depth_cb(self, msg: Image) -> None:
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _process_tick(self) -> None:
        if self.latest_rgb is None:
            return
        if self.reference_bgr is None:
            return

        current = self._segment_objects(self.latest_rgb, self.latest_depth)
        if not self.reference_detections:
            self.reference_detections = self._segment_objects(self.reference_bgr, None)
            self.get_logger().info(f'Initialized {len(self.reference_detections)} reference objects.')

        moved = self._find_moved_objects(current, self.reference_detections)
        if not moved:
            self.get_logger().info('Scene aligned with reference (within tolerance).')
            return

        # Prioritize biggest changed object first.
        moved.sort(key=lambda x: x.area_px, reverse=True)
        next_obj = moved[0]
        if next_obj.xyz_cam is None:
            self.get_logger().warn(f'No valid depth for "{next_obj.label}" detection.')
            return
        self._publish_point(next_obj)

    def _segment_objects(self, bgr: np.ndarray, depth: Optional[np.ndarray]) -> List[Detection2D3D]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        detections: List[Detection2D3D] = []
        for label, (lo, hi) in self.hsv_ranges.items():
            mask = cv2.inRange(hsv, lo, hi)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_contour_area_px:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                u = int(x + w / 2)
                v = int(y + h / 2)
                xyz = self._pixel_to_camera_xyz(u, v, depth)
                detections.append(
                    Detection2D3D(
                        label=label,
                        centroid_uv=(u, v),
                        bbox_xywh=(x, y, w, h),
                        area_px=area,
                        xyz_cam=xyz,
                    )
                )
        return detections

    def _pixel_to_camera_xyz(
        self, u: int, v: int, depth_img: Optional[np.ndarray]
    ) -> Optional[Tuple[float, float, float]]:
        if depth_img is None or self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return None
        if v < 0 or u < 0 or v >= depth_img.shape[0] or u >= depth_img.shape[1]:
            return None

        depth_patch = depth_img[max(v - 2, 0) : v + 3, max(u - 2, 0) : u + 3]
        valid = depth_patch[depth_patch > 0]
        if valid.size == 0:
            return None

        # Realsense depth image is usually uint16 in millimeters.
        z_m = float(np.median(valid))
        if depth_img.dtype == np.uint16:
            z_m = z_m / 1000.0
        x_m = (u - self.cx) * z_m / self.fx
        y_m = (v - self.cy) * z_m / self.fy
        return (x_m, y_m, z_m)

    def _find_moved_objects(
        self, current: List[Detection2D3D], reference: List[Detection2D3D]
    ) -> List[Detection2D3D]:
        moved: List[Detection2D3D] = []
        used_curr_idx = set()
        for ref in reference:
            best_idx = None
            best_dist = float('inf')
            for i, cur in enumerate(current):
                if i in used_curr_idx:
                    continue
                if cur.label != ref.label:
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
                moved.append(current[best_idx])
        return moved

    def _publish_point(self, detection: Detection2D3D) -> None:
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = 'camera_color_optical_frame'
        pt.point.x = detection.xyz_cam[0]
        pt.point.y = detection.xyz_cam[1]
        pt.point.z = detection.xyz_cam[2]
        self.pub_pose.publish(pt)
        self.get_logger().info(
            f'Publish move target "{detection.label}" at xyz='
            f'({pt.point.x:.3f}, {pt.point.y:.3f}, {pt.point.z:.3f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = TidyerPerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
