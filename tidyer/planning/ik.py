import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK, GetMotionPlan


class IKPlanner(Node):
    def __init__(self):
        super().__init__('ik_planner')
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')

    def compute_ik(self, current_joint_state, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0):
        return None  # TODO

    def plan_to_joints(self, target_joint_state):
        return None  # TODO


def main(args=None):
    rclpy.init(args=args)
    node = IKPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
