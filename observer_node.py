#!/usr/bin/env python3
import rclpy
import rclpy.node
import numpy as np
import rcl_interfaces.msg

import std_msgs.msg
import tmr4243_interfaces.msg

from luenberger_obs import LuenbergerObserver
import sensor_msgs.msg
from rclpy.qos import qos_profile_sensor_data

class Observer(rclpy.node.Node):
    TASK_DEADRECKONING = 'deadreckoning'
    TASK_LUENBERGER = 'luenberger'
    TASK_LIST = [TASK_DEADRECKONING, TASK_LUENBERGER]

    def __init__(self):
        super().__init__('cse_observer')

        self.subs = {}
        self.pubs = {}

        self.subs["tau"] = self.create_subscription(
            std_msgs.msg.Float32MultiArray, '/tmr4243/state/tau', self.tau_callback, qos_profile_sensor_data
        )
        self.subs["eta"] = self.create_subscription(
            std_msgs.msg.Float32MultiArray, '/tmr4243/state/eta', self.eta_callback, qos_profile_sensor_data
        )
        self.pubs['observer'] = self.create_publisher(
            tmr4243_interfaces.msg.Observer, '/tmr4243/observer/eta', 1
        )

        self.pubs["dr"] = self.create_publisher(std_msgs.msg.Bool, "/tmr4243/observer/dead_reckoning", 1)

        self.manual_deadreckoning = False
        self.last_btn = 0
        self.subs["joy"] = self.create_subscription(
            sensor_msgs.msg.Joy, "/joy", self.joy_callback, 10
        )
        # ---------------- Parameters ----------------
        self.task = Observer.TASK_LUENBERGER
        self.declare_parameter(
            'task',
            self.task,
            rcl_interfaces.msg.ParameterDescriptor(
                description="Task",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_STRING,
                read_only=False,
                additional_constraints=f"Allowed values: {' '.join(Observer.TASK_LIST)}"
            )
        )
        self.omega_c = 4.0
        self.declare_parameter(
            'L1',
            [self.omega_c, self.omega_c, self.omega_c],
            rcl_interfaces.msg.ParameterDescriptor(
                description="L1 diagonal entries",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY
            )
        )
        self.declare_parameter(
            'L2',
            [2.25*self.omega_c, 2.25*self.omega_c, 1.75*self.omega_c],
            rcl_interfaces.msg.ParameterDescriptor(
                description="L2 diagonal entries",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY
            )
        )
        self.declare_parameter(
            'L3',
            [0.225*self.omega_c, 0.225*self.omega_c, 0.175*self.omega_c],
            rcl_interfaces.msg.ParameterDescriptor(
                description="L3 diagonal entries",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY
            )
        )
        
        self.last_eta = np.zeros((3, 1), dtype=float)
        self.last_tau = np.zeros((3, 1), dtype=float)
        
        self.eta_updated = False
        self.missed_eta = 0

        self.obs_core = LuenbergerObserver()

        self.timer_period = 0.1
        self.last_time = self.get_clock().now()

        self.get_logger().info("Observer node started.")
        self.observer_runner = self.create_timer(self.timer_period, self.observer_loop)

        self.declare_parameter("drop_eta", False)  # for auto DR testing

    def observer_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if self.eta_updated:
            self.missed_eta = 0
        else:
            self.missed_eta += 1
        self.eta_updated = False

        auto_deadreckoning = (self.missed_eta >= 4)


        # Read parameters (can be tuned online via ros2 param set)
        task = self.get_parameter('task').get_parameter_value().string_value

        L1_value = self.get_parameter('L1').get_parameter_value().double_array_value
        L2_value = self.get_parameter('L2').get_parameter_value().double_array_value
        L3_value = self.get_parameter('L3').get_parameter_value().double_array_value

        L1 = np.diag(L1_value)
        L2 = np.diag(L2_value)
        L3 = np.diag(L3_value)

        #dead_reckoning = (task == Observer.TASK_DEADRECKONING)
        dead_reckoning = self.manual_deadreckoning or auto_deadreckoning

        m = std_msgs.msg.Bool()
        m.data = bool(dead_reckoning)
        self.pubs["dr"].publish(m)
        
        eta_hat, nu_hat, bias_hat = self.obs_core.step(
            eta_meas=self.last_eta,
            tau_meas=self.last_tau,
            L1=L1, L2=L2, L3=L3,
            dt=max(dt, 1e-6),
            dead_reckoning=dead_reckoning
        )

        obs = tmr4243_interfaces.msg.Observer()
        obs.eta = eta_hat.flatten().tolist()
        obs.nu = nu_hat.flatten().tolist()
        obs.bias = bias_hat.flatten().tolist()
        self.pubs['observer'].publish(obs)

    def tau_callback(self, msg: std_msgs.msg.Float32MultiArray):
        self.last_tau = np.array([msg.data], dtype=float).T

    def eta_callback(self, msg: std_msgs.msg.Float32MultiArray):
        if self.get_parameter("drop_eta").get_parameter_value().bool_value:
            return
        self.last_eta = np.array([msg.data], dtype=float).T
        self.eta_updated = True
    
    def joy_callback(self, msg):
        btn = int(msg.buttons[0])  # X for manual DR
        if btn == 1 and self.last_btn == 0:
            self.manual_deadreckoning = not self.manual_deadreckoning
            self.get_logger().info(f"Manual DR: {self.manual_deadreckoning}")
        self.last_btn = btn


def main():
    rclpy.init()
    node = Observer()
    rclpy.spin(node)


if __name__ == '__main__':
    main()