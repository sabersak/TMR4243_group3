#!/usr/bin/env python3
import numpy as np
import rclpy
import rclpy.node
import std_msgs.msg
import sensor_msgs.msg
import rcl_interfaces.msg

from joystick_mapping import JoystickMapping
from joystick_simple import joystick_simple
from joystick_force_basin_relative import joystick_force_basin_relative
from joystick_force_body_relative import joystick_force_body_relative


class JoystickControl(rclpy.node.Node):
    TASK_SIMPLE = 'simple'
    TASK_BASIN  = 'basin'
    TASK_BODY   = 'body'
    TASKS = [TASK_SIMPLE, TASK_BODY, TASK_BASIN]

    def __init__(self):
        super().__init__('tmr4243_joystick_control_node')

        self.sub_joy = self.create_subscription(
            sensor_msgs.msg.Joy, '/joy', self.joy_callback, 10)

        self.sub_eta = self.create_subscription(
            std_msgs.msg.Float32MultiArray, '/tmr4243/state/eta', self.eta_callback, 10)

        # publish u (5) for simple mode
        self.pub_u = self.create_publisher(
            std_msgs.msg.Float32MultiArray, '/tmr4243/command/u', 10)

        # publish tau (3) for body/basin
        self.pub_tau = self.create_publisher(
            std_msgs.msg.Float32MultiArray, '/tmr4243/command/tau', 10)

        self.task = JoystickControl.TASK_SIMPLE
        self.declare_parameter(
            'task',
            self.task,
            rcl_interfaces.msg.ParameterDescriptor(
                description="Task",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_STRING,
                additional_constraints=f"Allowed values: {' '.join(JoystickControl.TASKS)}"
            )
        )

        # basin reference heading (optional)
        self.psi_mc = 0.0
        self.declare_parameter('psi_mc', self.psi_mc)

        self.joystick_mapping = JoystickMapping()
        joystick_params = [
            'LEFT_STICK_HORIZONTAL', 'LEFT_STICK_VERTICAL',
            'RIGHT_STICK_HORIZONTAL', 'RIGHT_STICK_VERTICAL',
            'LEFT_TRIGGER', 'RIGHT_TRIGGER',
            'A_BUTTON', 'B_BUTTON', 'X_BUTTON', 'Y_BUTTON'
        ]
        for p in joystick_params:
            self.declare_parameter(p, getattr(self.joystick_mapping, p))
        for p in joystick_params:
            setattr(self.joystick_mapping, p, int(self.get_parameter(p).value))

        self.last_eta = None
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        self.task = self.get_parameter('task').get_parameter_value().string_value
        self.psi_mc = float(self.get_parameter('psi_mc').value)
        self.get_logger().info(f"task={self.task}", throttle_duration_sec=1.0)

    def eta_callback(self, msg):
        if msg.data and len(msg.data) == 3:
            self.last_eta = np.array(msg.data, dtype=float)
        else:
            self.last_eta = None

    def joy_callback(self, msg):
        if self.task == JoystickControl.TASK_SIMPLE:
            # u is (5,1)
            u = joystick_simple(msg, self.joystick_mapping)
            if np.size(u) != 5:
                self.get_logger().warn("joystick_simple must return 5 values.")
                return
            out = std_msgs.msg.Float32MultiArray()
            out.data = [float(x) for x in np.asarray(u).reshape(-1)]
            self.pub_u.publish(out)
            return

        # body/basin publish tau (3,1)
        if self.task == JoystickControl.TASK_BODY:
            tau = joystick_force_body_relative(msg, self.joystick_mapping)

        elif self.task == JoystickControl.TASK_BASIN:
            if self.last_eta is None:
                self.get_logger().warn("No eta yet; cannot basin-relative.", throttle_duration_sec=1.0)
                return
            tau = joystick_force_basin_relative(msg, self.last_eta, self.joystick_mapping, psi_mc=self.psi_mc)

        else:
            self.get_logger().warn(f"Unknown task={self.task}")
            return

        if np.size(tau) != 3:
            self.get_logger().warn("tau must be 3 values.")
            return

        out = std_msgs.msg.Float32MultiArray()
        out.data = [float(x) for x in np.asarray(tau).reshape(-1)]
        self.pub_tau.publish(out)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(JoystickControl())
    rclpy.shutdown()

if __name__ == '__main__':
    main()