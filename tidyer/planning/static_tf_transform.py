import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


def main():
    rclpy.init()
    node = Node('tidyer_tf')
    br = StaticTransformBroadcaster(node)
    t = TransformStamped()
    t.header.frame_id = 'wrist_3_link'
    t.child_frame_id = 'camera_color_optical_frame'
    t.transform.translation.x = -0.04
    t.transform.translation.y = 0.13
    t.transform.translation.z = -0.025 #0.00

    def tick():
        t.header.stamp = node.get_clock().now().to_msg()
        br.sendTransform(t)

    node.create_timer(0.05, tick)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
