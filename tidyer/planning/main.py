from typing import List, Optional, Union

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseArray
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory

from tidyer.planning.ik import IKPlanner

UR_JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]
# Ordered to match UR_JOINT_NAMES (pan, lift, elbow, w1, w2, w3).
DEFAULT_JOINTS = [
    4.751084327697754,
    -1.2080865663341065,
    -2.2311320304870605,
    -1.255601243381836,
    1.5683932304382324,
    -3.1400280634509485,
]

# Lab5 pattern: queue entries are either a JointState (planned move via
# plan_to_joints) or the literal string 'toggle_grip' (gripper service).
Job = Union[JointState, str]


class UR7e_CubeGrasp(Node):
    def __init__(self) -> None:
        super().__init__('cube_grasp')
        self.active_locations: List[Pose] = []
        self.overlap_threshold: float = 0.05

        ## TODO: CHECK if these are the right intermediate locations for the camera setup, placeholder for now
        self.intermediate_locations: List[Pose] = [
            self._pose_from_xyz_yaw(0.30, 0.65, -0.16, 0.0),
            self._pose_from_xyz_yaw(0.36, 0.65, -0.16, 0.0),
            self._pose_from_xyz_yaw(0.42, 0.65, -0.16, 0.0),

        ]

        # Geometry params (base_link frame).
        self.declare_parameter('gripper_offset_m', 0.150)       # wrist_3_link to fingertip
        self.declare_parameter('finger_insertion_m', 0.015)     # TODO: tune empirically
        self.declare_parameter('approach_height_m', 0.035)      # extra clearance above pre-grasp

        self.gripper_offset_m = float(self.get_parameter('gripper_offset_m').value)
        self.finger_insertion_m = float(self.get_parameter('finger_insertion_m').value)
        self.approach_height_m = float(self.get_parameter('approach_height_m').value)

        self.create_subscription(PoseArray, '/pick_place_pair', self.pair_callback, 1)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
        )
        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.joint_state: Optional[JointState] = None
        self.ik_planner = IKPlanner()

        self.job_queue: List[Job] = []
        self.busy: bool = False

        self._startup_timer = self.create_timer(0.1, self._startup_move)

    @staticmethod
    def _default_joint_state() -> JointState:
        js = JointState()
        js.name = list(UR_JOINT_NAMES)
        js.position = list(DEFAULT_JOINTS)
        return js
    
    def _overlap_check(self, new_pose: Pose) -> Optional[Pose]:
        for existing in self.active_locations:
            dx = existing.position.x - new_pose.position.x
            dy = existing.position.y - new_pose.position.y
            dz = existing.position.z - new_pose.position.z
            dist = (dx**2 + dy**2 + dz**2)**0.5
            if dist < self.overlap_threshold:
                return existing 
        return None
    
    def _free_intermediate_location(self) -> Optional[Pose]:
        for loc in self.intermediate_locations:
            if not self._overlap_check(loc):
                return loc
        return None
    
    def _queue_pick_and_place(self, source: Pose, destination: Pose) -> bool:

        pre_pick = self._lift(
            source,
            self.gripper_offset_m + self.approach_height_m
        )

        grasp_pick = self._lift(
            source,
            self.gripper_offset_m - self.finger_insertion_m
        )

        pre_place = self._lift(
            destination,
            self.gripper_offset_m + self.approach_height_m
        )

        place_pose = self._lift(
            destination,
            self.gripper_offset_m - self.finger_insertion_m
        )

        pre_pick_js = self._ik_or_abort(
            self.joint_state,
            pre_pick,
            'pre-pick'
        )

        if pre_pick_js is None:
            return False

        grasp_js = self._ik_or_abort(
            pre_pick_js,
            grasp_pick,
            'grasp'
        )

        if grasp_js is None:
            return False

        pre_place_js = self._ik_or_abort(
            grasp_js,
            pre_place,
            'pre-place'
        )

        if pre_place_js is None:
            return False

        place_js = self._ik_or_abort(
            pre_place_js,
            place_pose,
            'place'
        )

        if place_js is None:
            return False

        self.job_queue.extend([
            pre_pick_js,
            grasp_js,
            'toggle_grip',

            pre_pick_js,

            pre_place_js,
            place_js,
            'toggle_grip',

            pre_place_js,
        ])

        return True

    def _startup_move(self) -> None:
        # Wait for /joint_states so we can tell whether we're already home.
        if self.joint_state is None:
            return
        self._startup_timer.cancel()
        if self._at_default_pose():
            self.get_logger().info('Already at default joint position; skipping startup move.')
            self.busy = False
            return
        self.get_logger().info('Moving to default joint position at startup.')
        self.job_queue.append(self._default_joint_state())
        self.busy = True
        self.execute_jobs()

    def _at_default_pose(self, tol: float = 0.05) -> bool:
        if self.joint_state is None:
            return False
        pos_by_name = dict(zip(self.joint_state.name, self.joint_state.position))
        for name, target in zip(UR_JOINT_NAMES, DEFAULT_JOINTS):
            if name not in pos_by_name or abs(pos_by_name[name] - target) > tol:
                return False
        return True

    def joint_state_callback(self, msg: JointState) -> None:
        self.joint_state = msg

    def pair_callback(self, msg: PoseArray) -> None:
        self._run_pair(msg)

    def _run_pair(self, msg: PoseArray) -> None:
        self.get_logger().info("RUNNING THE PAIR CALLBACK")
        self.get_logger().info(f"Busy: {self.busy}")
        if self.busy:
            self.get_logger().info('Pick/place in progress; ignoring new pair.')
            return
        if self.joint_state is None:
            self.get_logger().warn('No joint state yet; cannot proceed.')
            return
        if len(msg.poses) < 2:
            self.get_logger().error(f'Expected 2 poses (pick, place), got {len(msg.poses)}.')
            return

        pick = msg.poses[0]
        place = msg.poses[1]
        place_conflict = self._overlap_check(place)

        if place_conflict is not None:

            self.get_logger().info(
                'Place location occupied. Relocating blocking object.'
            )

            intermediate = self._free_intermediate_location()

            if intermediate is None:
                self.get_logger().error(
                    'No free intermediate locations available.'
                )
                return
            
            success = self._queue_pick_and_place(place_conflict, intermediate)
            if not success:
                return

            self.active_locations.remove(place_conflict)
            self.active_locations.append(intermediate)
        success = self._queue_pick_and_place( pick, place)

        if not success:
            return
        pick_conflict = self._overlap_check(pick)

        if pick_conflict is not None:
            self.active_locations.remove(pick_conflict)

        self.active_locations.append(place)

        
        self.job_queue.append(self._default_joint_state())
        self.busy = True
        self.execute_jobs()

    @staticmethod
    def _pose_from_xyz_yaw(x: float, y: float, z: float, yaw_rad: float) -> Pose:
        qx, qy, qz, qw = R.from_euler('xyz', [np.pi, 0.0, yaw_rad]).as_quat()
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.x = float(qx)
        pose.orientation.y = float(qy)
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)
        return pose

    def _ik_or_abort(self, seed: JointState, pose: Pose, label: str) -> Optional[JointState]:
        result = self.ik_planner.compute_ik(
            seed,
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
        )
        if result is None:
            self.get_logger().error(f'IK failed for {label}; aborting cycle.')
        return result

    @staticmethod
    def _lift(pose: Pose, dz: float) -> Pose:
        out = Pose()
        out.position.x = pose.position.x
        out.position.y = pose.position.y
        out.position.z = pose.position.z + dz
        out.orientation = pose.orientation
        return out

    def execute_jobs(self) -> None:
        if not self.job_queue:
            self.get_logger().info('Cycle complete; back at observation pose. Press "c" for next.')
            self.busy = False

            return

        self.get_logger().info(f'Executing job queue, {len(self.job_queue)} jobs remaining.')
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, JointState):
            self._run_planned_move(next_job)
        elif next_job == 'toggle_grip':
            self._toggle_gripper()
        else:
            self.get_logger().error(f'Unknown job type: {type(next_job)}')
            self.execute_jobs()

    def _run_planned_move(self, target_joint_state: JointState) -> None:
        traj = self.ik_planner.plan_to_joints(target_joint_state)
        if traj is None:
            self.get_logger().error('Joint plan failed; aborting cycle.')
            self.job_queue.clear()
            self.busy = False
            return
        self._execute_joint_trajectory(traj.joint_trajectory)

    def _toggle_gripper(self) -> None:
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                'Gripper service /toggle_gripper not available; aborting cycle.'
            )
            self.job_queue.clear()
            self.busy = False
            return
        future = self.gripper_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done():
            self.get_logger().error('Gripper service call timed out; aborting cycle.')
            self.job_queue.clear()
            self.busy = False
            return
        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()

    def _execute_joint_trajectory(self, joint_traj: JointTrajectory) -> None:
        self.get_logger().info('Waiting for controller action server...')
        if not self.exec_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                'Controller action server /scaled_joint_trajectory_controller/'
                'follow_joint_trajectory not available; is the UR driver running?'
            )
            self.job_queue.clear()
            self.busy = False
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'send_goal failed: {exc}')
            self.job_queue.clear()
            self.busy = False
            return
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Trajectory goal rejected.')
            self.job_queue.clear()
            self.busy = False
            return
        self.get_logger().info('Executing trajectory...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future) -> None:
        try:
            future.result().result
            self.get_logger().info('Trajectory complete.')
            self.execute_jobs()
        except Exception as exc:
            self.get_logger().error(f'Trajectory execution failed: {exc}')
            self.job_queue.clear()
            self.busy = False


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
