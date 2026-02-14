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
        self.pub_u = self.create_publisher(Float32MultiArray, 'tmr4243/command/u', 10)

    def cb(self, msg):
        tau_cmd = np.array(msg.data, dtype=float)
        F_cmd, alpha_cmd, u_cmd = self.ta.allocate(tau_cmd)
        #print(f"tau_cmd={tau_cmd}  u_cmd={u_cmd}  F_cmd={F_cmd}  alpha_cmd={alpha_cmd}")

        u = np.array([u_cmd[2],u_cmd[0], u_cmd[1], alpha_cmd[0], alpha_cmd[1]]).T
        
        # Publish
        u_msg = Float32MultiArray()
        u_msg.data = np.asarray(u, dtype=float).flatten().tolist()
        self.pub_u.publish(u_msg)

        self.get_logger().info(
            f"tau={tau_cmd}  u={np.asarray(u).round(3)}  "
            f"F={np.asarray(F_cmd).round(3)}  alpha={np.asarray(alpha_cmd).round(3)}"
        )


def main():
    rclpy.init()
    rclpy.spin(TANode())
    rclpy.shutdown()
