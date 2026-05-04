import sys

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetMotionPlan, GetPositionIK
from shape_msgs.msg import SolidPrimitive


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

    def plan_to_pose(self, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0,
                     planning_time_s: float = 5.0, attempts: int = 10):
        """Plan to the given EE pose. Returns moveit_msgs/RobotTrajectory or None."""
        if not self.plan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/plan_kinematic_path service not available.')
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

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.001]

        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv.primitive_poses.append(pose.pose)

        position_c = PositionConstraint()
        position_c.header.frame_id = self.BASE_FRAME
        position_c.link_name = self.EE_LINK
        position_c.constraint_region = bv
        position_c.weight = 1.0

        orientation_c = OrientationConstraint()
        orientation_c.header.frame_id = self.BASE_FRAME
        orientation_c.link_name = self.EE_LINK
        orientation_c.orientation = pose.pose.orientation
        orientation_c.absolute_x_axis_tolerance = 0.01
        orientation_c.absolute_y_axis_tolerance = 0.01
        orientation_c.absolute_z_axis_tolerance = 0.01
        orientation_c.weight = 1.0

        goal = Constraints()
        goal.position_constraints.append(position_c)
        goal.orientation_constraints.append(orientation_c)

        plan_req = MotionPlanRequest()
        plan_req.group_name = self.GROUP_NAME
        plan_req.num_planning_attempts = int(attempts)
        plan_req.allowed_planning_time = float(planning_time_s)
        plan_req.max_velocity_scaling_factor = 0.3
        plan_req.max_acceleration_scaling_factor = 0.3
        # Empty start_state with is_diff=True → use move_group's current state monitor.
        plan_req.start_state = RobotState()
        plan_req.start_state.is_diff = True
        plan_req.goal_constraints.append(goal)

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

    def compute_ik(self, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0,
                   timeout_s: float = 1.0):
        """Return joint positions for the given EE pose, or None on failure."""
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
        req.ik_request.robot_state.is_diff = True
        req.ik_request.timeout = Duration(seconds=timeout_s).to_msg()
        req.ik_request.avoid_collisions = True

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
        return list(resp.solution.joint_state.position)


def main(args=None):
    rclpy.init(args=args)
    node = IKPlanner()

    node.get_logger().info('Testing IK computation...')
    joint_positions = node.compute_ik(0.125, 0.611, 0.423)

    if joint_positions is None:
        node.get_logger().error('IK computation returned None.')
        sys.exit(1)

    if len(joint_positions) < 6:
        node.get_logger().error('IK returned fewer than 6 joints — likely incorrect.')
        sys.exit(1)

    node.get_logger().info(f'IK check passed: {list(joint_positions)}')

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
