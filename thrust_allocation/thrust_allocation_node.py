#!/usr/bin/env python3
import numpy as np
import rclpy
import rclpy.node
import std_msgs.msg
import rcl_interfaces.msg

from thrust_allocator import ThrustAllocator

class ThrustAllocation(rclpy.node.Node):
    ALLOC_FIXED = 'fixed'     # Task 3.4 (Eq. 6-7)
    ALLOC_VARYING = 'varying' # Task 3.5 (Eq. 11-12)

    def __init__(self):
        super().__init__("tmr4243_thrust_allocation_node")

        self.alloc = ThrustAllocator()
        self.alloc.setup_allocation_matrix()

        self.declare_parameter(
            'allocation_mode', ThrustAllocation.ALLOC_VARYING,
            rcl_interfaces.msg.ParameterDescriptor(
                description="fixed (Task3.4) or varying (Task3.5)",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_STRING
            )
        )
        self.declare_parameter('alpha_fixed_1', 0.0)
        self.declare_parameter('alpha_fixed_2', 0.0)

        self.sub_tau = self.create_subscription(
            std_msgs.msg.Float32MultiArray,
            '/tmr4243/command/tau',
            self.tau_cmd_callback,
            10
        )

        self.pub_u = self.create_publisher(
            std_msgs.msg.Float32MultiArray,
            '/tmr4243/command/u',
            10
        )

        self.last_tau = np.zeros(3, dtype=float)
        self.get_logger().info("Thrust allocation node started.")
        self.timer = self.create_timer(0.1, self.timer_callback) 
        

    def tau_cmd_callback(self, msg):
        if msg.data and len(msg.data) >= 3:
            self.last_tau = np.array(msg.data[:3], dtype=float)

    def timer_callback(self):
        mode = self.get_parameter('allocation_mode').get_parameter_value().string_value.lower()

        tau = self.last_tau

        if mode == ThrustAllocation.ALLOC_FIXED:
            a1 = float(self.get_parameter('alpha_fixed_1').value)
            a2 = float(self.get_parameter('alpha_fixed_2').value)
            alpha_fixed = np.array([a1, a2], dtype=float)

            # allocate_task1 returns (F_cmd, u_cmd3, alpha_cmd, F_star, tau_err)
            F_cmd, u_cmd3, alpha_cmd, F_star, tau_err = self.alloc.allocate_task1(tau, alpha_fixed)
            u1, u2, u3 = u_cmd3
            # publish fixed angles explicitly (Task 3.4)
            a1_pub, a2_pub = alpha_fixed

        else:
            # allocate_task2 returns (F_cmd, alpha_cmd, u_cmd3, f_star, tau_err)
            F_cmd, alpha_cmd, u_cmd3, f_star, tau_err = self.alloc.allocate_task2(tau)
            u1, u2, u3 = u_cmd3
            a1_pub, a2_pub = alpha_cmd

        # Pack for simulator: u = [u0,u1,u2,a1,a2]^T, where u0 is tunnel command (u3)
        u_vec = [float(u3), float(u1), float(u2), float(a1_pub), float(a2_pub)]

        msg_u = std_msgs.msg.Float32MultiArray()
        msg_u.data = u_vec
        self.pub_u.publish(msg_u)

        self.get_logger().info(f"mode={mode}", throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = ThrustAllocation()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()