import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped


def main(args=None):
    rclpy.init(args=args)
    node = Node('tidyer_pc')

    def cb(_msg: PointCloud2):
        pass

    node.create_subscription(PointCloud2, 'INSERT_TOPIC_NAME', cb, 10)
    node.create_publisher(PointStamped, '/cube_pose', 1)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
