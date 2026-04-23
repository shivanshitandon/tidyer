import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped


def main(args=None):
    rclpy.init(args=args)
    node = Node('tidyer_pick_place')

    def cb(_msg: PointStamped):
        pass

    node.create_subscription(PointStamped, '/cube_pose', cb, 1)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
