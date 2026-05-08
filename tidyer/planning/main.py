from typing import List, Optional, Union

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseArray
from rclpy.action import ActionClient
from rclpy.node import Node
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

        # Geometry params (base_link frame).
        self.declare_parameter('gripper_offset_m', 0.150)       # wrist_3_link to fingertip
        self.declare_parameter('finger_insertion_m', 0.045)     # TODO: tune empirically
        self.declare_parameter('approach_height_m', 0.015)      # extra clearance above pre-grasp
        self.declare_parameter('default_pose_tol_rad', 0.02)    # post-cycle verification
        self.declare_parameter('default_pose_max_retries', 2)

        self.gripper_offset_m = float(self.get_parameter('gripper_offset_m').value)
        self.finger_insertion_m = float(self.get_parameter('finger_insertion_m').value)
        self.approach_height_m = float(self.get_parameter('approach_height_m').value)
        self.default_pose_tol_rad = float(self.get_parameter('default_pose_tol_rad').value)
        self.default_pose_max_retries = int(self.get_parameter('default_pose_max_retries').value)
        self._default_pose_retries = 0

        self.create_subscription(PoseArray, '/pick_place_pair', self.pair_callback, 1)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
        )
        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')
        self.capture_cur_cli = self.create_client(Trigger, '/capture_current')

        self.create_service(Trigger, '/arm_auto_capture_chain', self._on_arm_auto_capture_chain)
        self.create_service(Trigger, '/disarm_auto_capture_chain', self._on_disarm_auto_capture_chain)

        self.joint_state: Optional[JointState] = None
        self.ik_planner = IKPlanner()

        self.job_queue: List[Job] = []
        self.busy: bool = False
        self.auto_capture_chain: bool = False

        self._startup_timer = self.create_timer(0.1, self._startup_move)

    def _on_arm_auto_capture_chain(self, request, response):
        self.auto_capture_chain = True
        response.success = True
        response.message = 'Auto capture chain armed.'
        return response

    def _on_disarm_auto_capture_chain(self, request, response):
        self.auto_capture_chain = False
        response.success = True
        response.message = 'Auto capture chain disarmed.'
        return response

    def _stop_auto_capture_chain(self, reason: str) -> None:
        if not self.auto_capture_chain:
            return
        self.auto_capture_chain = False
        self.get_logger().info(f'Auto capture chain stopped: {reason}')

    def _schedule_followup_capture_after_cycle(self) -> None:
        if not self.auto_capture_chain:
            return
        if not self.capture_cur_cli.wait_for_service(timeout_sec=2.0):
            self._stop_auto_capture_chain('/capture_current not available.')
            return
        fut = self.capture_cur_cli.call_async(Trigger.Request())
        fut.add_done_callback(self._on_followup_capture_done)

    def _on_followup_capture_done(self, future) -> None:
        if not self.auto_capture_chain:
            return
        try:
            response = future.result()
        except Exception as exc:
            self._stop_auto_capture_chain(f'capture_current error: {exc}')
            return
        if not response.success:
            self._stop_auto_capture_chain(response.message)
            return
        if 'Scene aligned with reference' in response.message:
            self.get_logger().info(response.message)
            self.auto_capture_chain = False
            return

    @staticmethod
    def _default_joint_state() -> JointState:
        js = JointState()
        js.name = list(UR_JOINT_NAMES)
        js.position = list(DEFAULT_JOINTS)
        return js

    def _startup_move(self) -> None:
        # Wait for /joint_states so we can tell whether we're already home.
        if self.joint_state is None:
            return
        self._startup_timer.cancel()
        if self._at_default_pose(tol=0.05):
            self.get_logger().info('Already at default joint position; skipping startup move.')
            self.busy = False
            return
        self.get_logger().info('Moving to default joint position at startup.')
        self.job_queue.append(self._default_joint_state())
        self.busy = True
        self.execute_jobs()

    def _at_default_pose(self, tol: Optional[float] = None) -> bool:
        if self.joint_state is None:
            return False
        if tol is None:
            tol = self.default_pose_tol_rad
        pos_by_name = dict(zip(self.joint_state.name, self.joint_state.position))
        for name, target in zip(UR_JOINT_NAMES, DEFAULT_JOINTS):
            if name not in pos_by_name or abs(pos_by_name[name] - target) > tol:
                return False
        return True

    def joint_state_callback(self, msg: JointState) -> None:
        self.joint_state = msg

    def pair_callback(self, msg: PoseArray) -> None:
        print("RUNNING THE PAIR CALLBACK")
        self.get_logger().info("RUNNING THE PAIR CALLBACK")
        self.get_logger().info(f"Busy: {self.busy}")
        if self.busy:
            self.get_logger().info('Pick/place in progress; ignoring new pair.')
            return
        if self.joint_state is None:
            self.get_logger().warn('No joint state yet; cannot proceed.')
            self._stop_auto_capture_chain('no joint state.')
            return
        if len(msg.poses) < 2:
            self.get_logger().error(f'Expected 2 poses (pick, place), got {len(msg.poses)}.')
            self._stop_auto_capture_chain('invalid PoseArray.')
            return

        pick = msg.poses[0]
        place = msg.poses[1]

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
            self._stop_auto_capture_chain('IK failed at pre-pick.')
            return
        grasp_js = self._ik_or_abort(pre_pick_js, grasp_pick, 'grasp')
        if grasp_js is None:
            self._stop_auto_capture_chain('IK failed at grasp.')
            return
        pre_place_js = self._ik_or_abort(grasp_js, pre_place, 'pre-place')
        if pre_place_js is None:
            self._stop_auto_capture_chain('IK failed at pre-place.')
            return
        place_js = self._ik_or_abort(pre_place_js, place_pose, 'place')
        if place_js is None:
            self._stop_auto_capture_chain('IK failed at place.')
            return

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
            # Verify we actually returned to default — controller can declare a
            # trajectory complete short of the goal (e.g. protective stop, IK
            # commanded a near-home but not-home state). Re-issue if needed.
            if self._at_default_pose():
                self._default_pose_retries = 0
                if self.auto_capture_chain:
                    self.get_logger().info(
                        'Cycle complete; verified at default pose. Scheduling next capture.'
                    )
                else:
                    self.get_logger().info(
                        'Cycle complete; verified at default pose. Press "c" for next.'
                    )
                self.busy = False
                self._schedule_followup_capture_after_cycle()
                return
            if self._default_pose_retries >= self.default_pose_max_retries:
                self.get_logger().error(
                    f'Failed to reach default pose after '
                    f'{self._default_pose_retries} retries; aborting cycle.'
                )
                self._default_pose_retries = 0
                self.busy = False
                self._stop_auto_capture_chain('default pose retries exhausted.')
                return
            self._default_pose_retries += 1
            self.get_logger().warn(
                f'Not at default pose after cycle (tol={self.default_pose_tol_rad} rad); '
                f're-queuing default move (retry '
                f'{self._default_pose_retries}/{self.default_pose_max_retries}).'
            )
            self.job_queue.append(self._default_joint_state())

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
            self._stop_auto_capture_chain('joint plan failed.')
            return
        self._execute_joint_trajectory(traj.joint_trajectory)

    def _toggle_gripper(self) -> None:
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                'Gripper service /toggle_gripper not available; aborting cycle.'
            )
            self.job_queue.clear()
            self.busy = False
            self._stop_auto_capture_chain('gripper service unavailable.')
            return
        future = self.gripper_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done():
            self.get_logger().error('Gripper service call timed out; aborting cycle.')
            self.job_queue.clear()
            self.busy = False
            self._stop_auto_capture_chain('gripper service timed out.')
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
            self._stop_auto_capture_chain('controller action server unavailable.')
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
            self._stop_auto_capture_chain(f'send_goal failed: {exc}')
            return
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Trajectory goal rejected.')
            self.job_queue.clear()
            self.busy = False
            self._stop_auto_capture_chain('trajectory goal rejected.')
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
            self._stop_auto_capture_chain(f'trajectory execution failed: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
