import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState


class IKPlanner(Node):
    def __init__(self):
        super().__init__('ik_planner')

        self.moveit = MoveItPy(node_name='ik_planner')
        self.arm = self.moveit.get_planning_component('ur_manipulator')
        self.robot_model = self.moveit.get_robot_model()

    def compute_ik(self, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0):
        """Return joint positions for the given end-effector pose, or None on failure."""
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        robot_state = RobotState(self.robot_model)
        success = robot_state.set_from_ik('ur_manipulator', pose.pose, 'wrist_3_link')
        if not success:
            self.get_logger().error('IK failed for the given pose.')
            return None

        self.get_logger().info('IK solution found.')
        return robot_state.get_joint_group_positions('ur_manipulator')

    def plan_to_pose(self, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0):
        """Plan a trajectory to the given end-effector pose. Returns the trajectory or None."""
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_stamped_msg=pose, pose_link='wrist_3_link')

        plan_result = self.arm.plan()
        if not plan_result:
            self.get_logger().error('Motion planning failed.')
            return None

        self.get_logger().info('Motion plan computed successfully.')
        return plan_result.trajectory

    def execute(self, trajectory):
        self.moveit.execute(trajectory, controllers=[])


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
