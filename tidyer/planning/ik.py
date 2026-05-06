import sys

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    RobotState,
)
from moveit_msgs.srv import GetMotionPlan, GetPositionIK


# moveit_msgs/MoveItErrorCodes.SUCCESS
_MOVEIT_SUCCESS = 1


class IKPlanner(Node):
    GROUP_NAME = 'ur_manipulator'
    EE_LINK = 'wrist_3_link'
    BASE_FRAME = 'base_link'

    def __init__(self):
        super().__init__('ik_planner')
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')

    # -----------------------------------------------------------
    # IK: Cartesian EE pose + seed joint configuration -> JointState.
    # Default quaternion (0, 1, 0, 0) is the lab5 top-down pose.
    # Pass a yaw-rotated quaternion to grasp blocks at arbitrary headings.
    # -----------------------------------------------------------
    def compute_ik(self, current_joint_state: JointState, x, y, z,
                   qx=0.0, qy=1.0, qz=0.0, qw=0.0,
                   timeout_s: float = 1.0):
        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/compute_ik service not available.')
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.BASE_FRAME
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.GROUP_NAME
        req.ik_request.ik_link_name = self.EE_LINK
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = Duration(seconds=timeout_s).to_msg()
        # Explicit seed so chained IKs stay on a single branch (avoids 180 deg
        # wrist flips between waypoints when their yaws differ).
        req.ik_request.robot_state.joint_state = current_joint_state
        req.ik_request.robot_state.is_diff = False

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s + 2.0)
        if not future.done() or future.result() is None:
            self.get_logger().error('IK request did not complete.')
            return None

        resp = future.result()
        if resp.error_code.val != _MOVEIT_SUCCESS:
            self.get_logger().error(f'IK failed (MoveIt error code {resp.error_code.val}).')
            return None

        self.get_logger().info('IK solution found.')
        return resp.solution.joint_state

    # -----------------------------------------------------------
    # Plan motion to a joint configuration (lab5 pattern):
    # joint-space goal constraints, RRTConnect, returns a
    # moveit_msgs/RobotTrajectory or None.
    # -----------------------------------------------------------
    def plan_to_joints(self, target_joint_state: JointState,
                       planning_time_s: float = 5.0):
        if not self.plan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/plan_kinematic_path service not available.')
            return None

        plan_req = MotionPlanRequest()
        plan_req.group_name = self.GROUP_NAME
        plan_req.allowed_planning_time = float(planning_time_s)
        plan_req.planner_id = 'RRTConnectkConfigDefault'
        plan_req.max_velocity_scaling_factor = 0.3
        plan_req.max_acceleration_scaling_factor = 0.3
        # Empty start_state with is_diff=True -> use move_group's current state monitor.
        plan_req.start_state = RobotState()
        plan_req.start_state.is_diff = True

        goal_constraints = Constraints()
        for name, pos in zip(target_joint_state.name, target_joint_state.position):
            goal_constraints.joint_constraints.append(
                JointConstraint(
                    joint_name=name,
                    position=pos,
                    tolerance_above=0.01,
                    tolerance_below=0.01,
                    weight=1.0,
                )
            )
        plan_req.goal_constraints.append(goal_constraints)

        srv_req = GetMotionPlan.Request()
        srv_req.motion_plan_request = plan_req

        future = self.plan_client.call_async(srv_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=planning_time_s + 5.0)
        if not future.done() or future.result() is None:
            self.get_logger().error('Motion plan request did not complete.')
            return None

        resp = future.result()
        err = resp.motion_plan_response.error_code.val
        if err != _MOVEIT_SUCCESS:
            self.get_logger().error(f'Planning failed (MoveIt error code {err}).')
            return None

        self.get_logger().info('Motion plan computed successfully.')
        return resp.motion_plan_response.trajectory


def main(args=None):
    rclpy.init(args=args)
    node = IKPlanner()

    seed = JointState()
    seed.name = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
    ]
    seed.position = [4.722, -1.850, -1.425, -1.405, 1.593, -3.141]

    node.get_logger().info('Testing IK computation...')
    ik_result = node.compute_ik(seed, 0.125, 0.611, 0.423)
    if ik_result is None:
        node.get_logger().error('IK computation returned None.')
        sys.exit(1)
    if len(ik_result.position) < 6:
        node.get_logger().error('IK returned fewer than 6 joints — likely incorrect.')
        sys.exit(1)

    node.get_logger().info(f'IK check passed: {list(ik_result.position)}')

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
