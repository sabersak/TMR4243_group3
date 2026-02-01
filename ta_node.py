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

        # Publishers 
        self.pub_u = self.create_publisher(Float32MultiArray, 'u_cmd', 10)
        self.pub_alpha = self.create_publisher(Float32MultiArray, 'alpha_cmd', 10)
        self.pub_F = self.create_publisher(Float32MultiArray, 'F_cmd', 10)

    def cb(self, msg):
        tau_cmd = np.array(msg.data, dtype=float)
        F_cmd, alpha_cmd, u_cmd = self.ta.allocate(tau_cmd)
        #print(f"tau_cmd={tau_cmd}  u_cmd={u_cmd}  F_cmd={F_cmd}  alpha_cmd={alpha_cmd}")

        # Publish
        u_msg = Float32MultiArray()
        u_msg.data = np.asarray(u_cmd, dtype=float).flatten().tolist()
        self.pub_u.publish(u_msg)

        a_msg = Float32MultiArray()
        a_msg.data = np.asarray(alpha_cmd, dtype=float).flatten().tolist()
        self.pub_alpha.publish(a_msg)

        f_msg = Float32MultiArray()
        f_msg.data = np.asarray(F_cmd, dtype=float).flatten().tolist()
        self.pub_F.publish(f_msg)

        self.get_logger().info(
            f"tau={tau_cmd}  u={np.asarray(u_cmd).round(3)}  "
            f"F={np.asarray(F_cmd).round(3)}  alpha={np.asarray(alpha_cmd).round(3)}"
        )


def main():
    rclpy.init()
    rclpy.spin(TANode())
    rclpy.shutdown()
