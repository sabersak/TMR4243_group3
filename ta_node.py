import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .thrust_allocation import ThrustAllocator


class TANode(Node):
    def __init__(self):
        super().__init__('ta_node')
        self.ta = ThrustAllocator()

        self.create_subscription(
            Float32MultiArray,
            'tau_cmd',
            self.cb,
            10
        )

    def cb(self, msg):
        tau_cmd = np.array(msg.data, dtype=float)
        F_cmd, alpha_cmd, u_cmd = self.ta.allocate(tau_cmd)
        print(f"tau={tau_cmd}  u={u_cmd}  F={F_cmd}  alpha={alpha_cmd}")


def main():
    rclpy.init()
    rclpy.spin(TANode())
    rclpy.shutdown()
