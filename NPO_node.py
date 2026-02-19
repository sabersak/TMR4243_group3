#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from NPO import NPOObserver


class NPOObserverNode(Node):

    def __init__(self):
        super().__init__('npo_observer_node')

        self.obs = NPOObserver()

        self.eta = np.zeros(3)
        self.nu  = np.zeros(3)
        self.tau = np.zeros(3)
        self.got_eta = False
        self.got_nu  = False
        self.initialized = False

        self.create_subscription(
            Float32MultiArray,
            'tmr4243/states/eta',
            self.cb_eta,
            10
        )

        self.create_subscription(
            Float32MultiArray,
            'tmr4243/states/nu',
            self.cb_nu,
            10
        )

        self.create_subscription(
            Float32MultiArray,
            '/tau_cmd',
            self.cb_tau,
            10
        )

        self.pub = self.create_publisher(
            Float32MultiArray,
            '/tau_cmd_obs',
            10
        )

        self.dt = 0.01
        self.create_timer(self.dt, self.step)

    def cb_eta(self, msg):
        self.eta = np.array(msg.data[:3], dtype=float)
        self.got_eta = True

    def cb_nu(self, msg):
        self.nu = np.array(msg.data[:3], dtype=float)
        self.got_nu = True
        
    def cb_tau(self, msg):
        self.tau = np.array(msg.data[:3], dtype=float)

    def step(self):
        if (not self.initialized) and self.got_eta and self.got_nu:
            self.obs.reset(self.eta, self.nu)
            self.initialized = True

        if not self.initialized:
            return

        eta_hat, nu_hat, b_hat = self.obs.step(self.dt, self.eta, self.tau)

        out = Float32MultiArray()
        out.data = [float(b_hat[0]), float(b_hat[1]), float(b_hat[2])]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = NPOObserverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()