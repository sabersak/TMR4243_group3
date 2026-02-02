#!/usr/bin/env python3
#
# This file is part of CyberShip Enterpries Suite.
#
# CyberShip Enterpries Suite software is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CyberShip Enterpries Suite is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# CyberShip Enterpries Suite. If not, see <https://www.gnu.org/licenses/>.
#
# Maintainer: Emir Cem Gezer, Petter Hangerhageen
# Email: emir.cem.gezer@ntnu.no, emircem.gezer@gmail.com, me@emircem.com, petthang@stud.ntnu.no
# Year: 2024
# Copyright (C) 2024 NTNU Marine Cybernetics Laboratory

import rclpy
import rclpy.node
import std_msgs.msg
import numpy as np

from template_thrust_allocation.thruster_allocation import thruster_allocation


class ThrustAllocation(rclpy.node.Node):
    def __init__(self):
        super().__init__("tmr4243_thrust_allocation_node")

        self.pubs = {}
        self.subs = {}

        self.subs["tau_cmd"] = self.create_subscription(
            std_msgs.msg.Float32MultiArray, '/tmr4243/command/tau', self.tau_cmd_callback, 1)

        self.pubs["u_cmd"] = self.create_publisher(
            std_msgs.msg.Float32MultiArray, '/tmr4243/command/u', 1)

        self.last_tau = np.zeros((3, 1), dtype=float)

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):

        u_cmd = std_msgs.msg.Float32MultiArray()

        u_cmd.data = thruster_allocation(self.tau).flatten().tobytes()

        self.pubs["u_cmd"].publish(u_cmd)

    def tau_cmd_callback(self, msg):

        self.last_tau = np.array([msg.data], dtype=float).T


def main(args=None):
    # Initialize the node
    rclpy.init(args=args)

    node = ThrustAllocation()

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
