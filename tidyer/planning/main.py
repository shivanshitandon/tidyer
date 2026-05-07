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

# Hardcoded pick/place pairs in base_link from successful camera runs.
HARDCODED_PAIRS = [
    {
        'name': 'yellow_rectangle',
        'pick_xyz': (-0.03578039524852805, 0.5472794341780659, -0.16157448993247991),
        'pick_yaw': 0.6055507724605418,
        'place_xyz': (0.020677794333050198, 0.5387825324148371, -0.15546467628787453),
        'place_yaw': -0.054235710904864565,
    },
    {
        'name': 'blue_rectangle',
        'pick_xyz': (0.22096279925070234, 0.46760614064292017, -0.16398030039062095),
        'pick_yaw': -0.8411308775631364,
        'place_xyz': (0.1475358967402308, 0.5319506789729332, -0.163920298588218),
        'place_yaw': 0.21127379438372085,
    },
]

# Lab5 pattern: queue entries are either a JointState (planned move via
# plan_to_joints) or the literal string 'toggle_grip' (gripper service).
Job = Union[JointState, str]


class UR7e_CubeGrasp(Node):
    def __init__(self) -> None:
        super().__init__('cube_grasp')
        self.active_location: List[Pose] = []
        self.overlap_threshold: float = 0.05

        # Geometry params (base_link frame).
        self.declare_parameter('gripper_offset_m', 0.150)       # wrist_3_link to fingertip
        self.declare_parameter('finger_insertion_m', 0.015)     # TODO: tune empirically
        self.declare_parameter('approach_height_m', 0.035)      # extra clearance above pre-grasp
        self.declare_parameter('hardcoded_pair_index', 0)

        self.gripper_offset_m = float(self.get_parameter('gripper_offset_m').value)
        self.finger_insertion_m = float(self.get_parameter('finger_insertion_m').value)
        self.approach_height_m = float(self.get_parameter('approach_height_m').value)
        self.hardcoded_pair_index = int(self.get_parameter('hardcoded_pair_index').value)

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
        self.hardcoded_cycle_started: bool = False

        self._startup_timer = self.create_timer(0.1, self._startup_move)
        self._hardcoded_timer = self.create_timer(0.3, self._maybe_start_hardcoded_cycle)

    @staticmethod
    def _default_joint_state() -> JointState:
        js = JointState()
        js.name = list(UR_JOINT_NAMES)
        js.position = list(DEFAULT_JOINTS)
        return js
    
    def _overlap_check(self, new_pose: Pose) -> bool:
        for existing in self.active_location:
            dx = existing.position.x - new_pose.position.x
            dy = existing.position.y - new_pose.position.y
            dz = existing.position.z - new_pose.position.z
            dist = (dx**2 + dy**2 + dz**2)**0.5
            if dist < self.overlap_threshold:
                return True
        return False

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

    def pair_callback(self, _msg: PoseArray) -> None:
        self.get_logger().info('Ignoring /pick_place_pair input; planner is in hardcoded mode.')
        return

    def _run_pair(self, msg: PoseArray) -> None:
        print("RUNNING THE PAIR CALLBACK")
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
        if self._overlap_check(pick) or self._overlap_check(place):
            self.get_logger().warn('Received pick/place pair too close to active location; ignoring.')
            return

        pre_pick = self._lift(pick, self.gripper_offset_m + self.approach_height_m)
        grasp_pick = self._lift(pick, self.gripper_offset_m - self.finger_insertion_m)
        pre_place = self._lift(place, self.gripper_offset_m + self.approach_height_m)
        place_pose = self._lift(place, self.gripper_offset_m - self.finger_insertion_m)

        # Run IK for every waypoint up front, seed-chained: each call uses the
        # previous IK result as its seed so the planner stays on a consistent
        # IK branch across the pick->place transition (avoids wrist flips when
        # the two yaws differ).
        pre_pick_js = self._ik_or_abort(self.joint_state, pre_pick, 'pre-pick')
        if pre_pick_js is None:
            return
        grasp_js = self._ik_or_abort(pre_pick_js, grasp_pick, 'grasp')
        if grasp_js is None:
            return
        pre_place_js = self._ik_or_abort(grasp_js, pre_place, 'pre-place')
        if pre_place_js is None:
            return
        place_js = self._ik_or_abort(pre_place_js, place_pose, 'place')
        if place_js is None:
            return
        
        self.active_locations.append(pick)
        self.active_locations.append(place)

        self.job_queue = [
            pre_pick_js,
            grasp_js,
            'toggle_grip',
            pre_pick_js,
            pre_place_js,
            place_js,
            'toggle_grip',
            pre_place_js,
            self._default_joint_state(),
        ]
        self.busy = True
        self.execute_jobs()

    def _maybe_start_hardcoded_cycle(self) -> None:
        if self.hardcoded_cycle_started:
            return
        if self.busy or self.joint_state is None:
            return
        idx = max(0, min(self.hardcoded_pair_index, len(HARDCODED_PAIRS) - 1))
        pair = HARDCODED_PAIRS[idx]
        pick = self._pose_from_xyz_yaw(*pair['pick_xyz'], pair['pick_yaw'])
        place = self._pose_from_xyz_yaw(*pair['place_xyz'], pair['place_yaw'])
        msg = PoseArray()
        msg.poses = [pick, place]
        self.hardcoded_cycle_started = True
        self.get_logger().info(
            f"Hardcoded pair mode: running '{pair['name']}' (index={idx})."
        )
        self._run_pair(msg)

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

            if self.active_locations:
                self.get_logger().info('Clear active locations')
                self.active_locations.clear()   
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
