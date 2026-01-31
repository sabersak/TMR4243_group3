import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .keyboard_logic import KeyboardLogic


class KeyboardTauNode(Node):
    def __init__(self):
        super().__init__('keyboard_tau')
        self.pub = self.create_publisher(Float32MultiArray, 'tau_cmd', 10)
        self.kb = KeyboardLogic()
        self.create_timer(0.05, self.loop)

    def loop(self):
        tau_cmd = self.kb.get_tau_cmd()
        msg = Float32MultiArray()
        msg.data = tau_cmd.tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(KeyboardTauNode())
    rclpy.shutdown()
