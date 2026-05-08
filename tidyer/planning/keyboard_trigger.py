import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


HELP = (
    "\nTidyer keyboard trigger\n"
    "  d : capture DESK DEPTH (clear desk first; saves desk plane depth)\n"
    "  r : capture REFERENCE image (target scene)\n"
    "  c : capture CURRENT image (publish next pick/place pair)\n"
    "  s : toggle STACKING mode (skip displacement on contour overlap)\n"
    "  q : quit\n"
)


class KeyboardTrigger(Node):
    def __init__(self) -> None:
        super().__init__('tidyer_keyboard_trigger')
        self.cli_ref = self.create_client(Trigger, '/capture_reference')
        self.cli_cur = self.create_client(Trigger, '/capture_current')
        self.cli_desk = self.create_client(Trigger, '/capture_desk_depth')
        self.cli_stack = self.create_client(Trigger, '/toggle_stacking')
        self.get_logger().info(HELP)

    def call(self, client, name: str) -> None:
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'{name} service not available')
            return
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.done() and future.result() is not None:
            res = future.result()
            self.get_logger().info(f'{name}: success={res.success} message="{res.message}"')
        else:
            self.get_logger().warn(f'{name} call did not complete in time')


def _read_key(timeout_s: float) -> str:
    if select.select([sys.stdin], [], [], timeout_s)[0]:
        return sys.stdin.read(1)
    return ''


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTrigger()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            key = _read_key(0.1)
            if not key:
                rclpy.spin_once(node, timeout_sec=0.0)
                continue
            if key == 'd':
                node.call(node.cli_desk, '/capture_desk_depth')
            elif key == 'r':
                node.call(node.cli_ref, '/capture_reference')
            elif key == 'c':
                node.call(node.cli_cur, '/capture_current')
            elif key == 's':
                node.call(node.cli_stack, '/toggle_stacking')
            elif key == 'q':
                break
            else:
                node.get_logger().info(HELP)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
