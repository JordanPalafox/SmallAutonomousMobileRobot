import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math


class VelocityPublisher(Node):
    def __init__(self):
        super().__init__('velocity_publisher')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(1.0, self.publish_velocity)
        self.t = 0.0
        self.get_logger().info('Velocity publisher started')

    def publish_velocity(self):
        msg = Twist()
        msg.linear.x = 0.5 * math.sin(self.t)
        msg.angular.z = 0.3 * math.cos(self.t)
        self.pub.publish(msg)
        self.get_logger().info(
            f'Publishing -> linear.x: {msg.linear.x:.3f}  angular.z: {msg.angular.z:.3f}'
        )
        self.t += 0.5


def main():
    rclpy.init()
    node = VelocityPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
