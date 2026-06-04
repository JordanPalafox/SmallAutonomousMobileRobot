import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class VelocitySubscriber(Node):
    def __init__(self):
        super().__init__('velocity_subscriber')
        self.create_subscription(Twist, '/cmd_vel', self.callback, 10)
        self.get_logger().info('Velocity subscriber started — waiting for /cmd_vel...')

    def callback(self, msg: Twist):
        self.get_logger().info(
            f'Received  <- linear.x: {msg.linear.x:.3f}  angular.z: {msg.angular.z:.3f}'
        )


def main():
    rclpy.init()
    node = VelocitySubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
