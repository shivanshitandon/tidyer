from dataclasses import dataclass
from typing import List, Union

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseArray
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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


@dataclass
class PoseGoal:
    pose: Pose


@dataclass
class JointGoal:
    positions: List[float]
    time_from_start_s: float = 5.0


@dataclass
class GripGoal:
    pass


Job = Union[PoseGoal, JointGoal, GripGoal]


class UR7e_CubeGrasp(Node):
    def __init__(self) -> None:
        super().__init__('cube_grasp')

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

        self.joint_state: JointState = None
        self.ik_planner = IKPlanner()

        self.job_queue: List[Job] = []
        self.busy: bool = False

        self._startup_timer = self.create_timer(0.1, self._startup_move)

    def _startup_move(self) -> None:
        self._startup_timer.cancel()
        self.get_logger().info('Moving to default joint position at startup.')
        self.job_queue.append(JointGoal(DEFAULT_JOINTS))
        self.busy = True
        self.execute_jobs()

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
            return
        if len(msg.poses) < 2:
            self.get_logger().error(f'Expected 2 poses (pick, place), got {len(msg.poses)}.')
            return

        pick = msg.poses[0]
        place = msg.poses[1]

        pre_pick = self._lift(pick, self.gripper_offset_m + self.approach_height_m)
        grasp_pick = self._lift(pick, self.gripper_offset_m - self.finger_insertion_m)
        pre_place = self._lift(place, self.gripper_offset_m + self.approach_height_m)
        place_pose = self._lift(place, self.gripper_offset_m - self.finger_insertion_m)

        self.job_queue = [
            PoseGoal(pre_pick),
            PoseGoal(grasp_pick),
            GripGoal(),
            PoseGoal(pre_pick),
            PoseGoal(pre_place),
            PoseGoal(place_pose),
            GripGoal(),
            PoseGoal(pre_place),
            JointGoal(DEFAULT_JOINTS),
        ]
        self.busy = True
        self.execute_jobs()

    @staticmethod
    def _lift(pose: Pose, dz: float) -> Pose:
        out = Pose()
        out.position.x = pose.position.x
        out.position.y = pose.position.y
        out.position.z = pose.position.z - dz
        out.orientation = pose.orientation
        return out

    def execute_jobs(self) -> None:
        if not self.job_queue:
            self.get_logger().info('Cycle complete; back at observation pose. Press "c" for next.')
            self.busy = False
            return

        self.get_logger().info(f'Executing job queue, {len(self.job_queue)} jobs remaining.')
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, PoseGoal):
            self._run_pose_goal(next_job.pose)
        elif isinstance(next_job, JointGoal):
            self._run_joint_goal(next_job.positions, next_job.time_from_start_s)
        elif isinstance(next_job, GripGoal):
            self._toggle_gripper()
        else:
            self.get_logger().error(f'Unknown job type: {type(next_job)}')
            self.execute_jobs()

    def _run_pose_goal(self, pose: Pose) -> None:
        ori = pose.orientation
        traj = self.ik_planner.plan_to_pose(
            pose.position.x, pose.position.y, pose.position.z,
            ori.x, ori.y, ori.z, ori.w,
        )
        if traj is None:
            self.get_logger().error('Pose plan failed; skipping job.')
            self.execute_jobs()
            return
        self._execute_joint_trajectory(traj.joint_trajectory)

    def _run_joint_goal(self, positions: List[float], time_from_start_s: float) -> None:
        traj = JointTrajectory()
        traj.joint_names = list(UR_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = DurationMsg(sec=int(time_from_start_s),
                                            nanosec=int((time_from_start_s % 1) * 1e9))
        traj.points = [point]
        self._execute_joint_trajectory(traj)

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
