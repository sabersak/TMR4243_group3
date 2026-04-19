#!/usr/bin/env python3
import rclpy
import rclpy.node
import rcl_interfaces.msg
import tmr4243_interfaces.msg
import std_msgs.msg
import numpy as np

from PID_controller import PID_controller
from PD_FF_controller import PD_FF_controller
from backstepping_controller import backstepping_controller


class Controller(rclpy.node.Node):
    TASK_PD_FF_CONTROLLER = 'PD_FF_controller'
    TASK_PID_CONTROLLER = 'PID_controller'
    TASK_BACKSTEPPING_CONTROLLER = 'backstepping_controller'
    TASKS = [TASK_PD_FF_CONTROLLER, TASK_PID_CONTROLLER, TASK_BACKSTEPPING_CONTROLLER]

    def __init__(self):
        super().__init__("tmr4243_controller")

        self.pubs = {}
        self.subs = {}

        self.subs["reference"] = self.create_subscription(
            tmr4243_interfaces.msg.Reference,
            '/tmr4243/control/reference',
            self.received_reference,
            10
        )

        self.subs['observer'] = self.create_subscription(
            tmr4243_interfaces.msg.Observer,
            '/tmr4243/observer/eta',
            self.received_observer,
            10
        )

        self.pubs["tau_cmd"] = self.create_publisher(
            std_msgs.msg.Float32MultiArray,
            '/tmr4243/command/tau',
            1
        )

        #self.p_gain = [5.0, 10.0, 7.0]  # exp values
        #self.p_gain = [5.0, 12.0, 4.0] # PD simulation values
        self.p_gain = [5.0, 8.0, 6.5] # PID simulation values
        
        self.declare_parameter(
            "p_gain",
            self.p_gain,
            rcl_interfaces.msg.ParameterDescriptor(
                description="Proportional gain diag entries [.,.,.]",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY,
                read_only=False
            )
        )

        #self.i_gain = [0.2, 0.2, 0.1] # exp values
        self.i_gain = [0.1, 0.12, 0.05] # PID simulation values

        self.declare_parameter(
            "i_gain",
            self.i_gain,
            rcl_interfaces.msg.ParameterDescriptor(
                description="Integral gain diag entries [.,.,.]",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY,
                read_only=False
            )
        )

        #self.d_gain = [14.0, 16.0, 6.0] # exp values
        #self.d_gain = [16.0, 19.0, 6.0]  # PD simulation values
        self.d_gain = [15.0, 23.0, 10.0]  # PID simulation values

        self.declare_parameter(
            "d_gain",
            self.d_gain,
            rcl_interfaces.msg.ParameterDescriptor(
                description="Derivative gain diag entries [.,.,.]",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY,
                read_only=False
            )
        )

        #self.k1_gain = [0.25, 0.30, 0.20]   # exp values
        self.k1_gain = [0.7, 0.8, 0.9]   # backstepping simulation values
        #self.k1_gain = [0.7, 0.3, 1.0]
        self.declare_parameter(
            "k1_gain",
            self.k1_gain,
            rcl_interfaces.msg.ParameterDescriptor(
                description="K1 diag entries [.,.,.]",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY,
                read_only=False
            )
        )

        #self.k2_gain = [7.0, 8.5, 4.2]     # exp values
        self.k2_gain = [3.5, 3.5, 4.0]  # backstepping simulation values
        #self.k2_gain = [4.0, 4.3, 4.5]

        self.declare_parameter(
            "k2_gain",
            self.k2_gain,
            rcl_interfaces.msg.ParameterDescriptor(
                description="K2 diag entries [.,.,.]",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_DOUBLE_ARRAY,
                read_only=False
            )
        )

        #self.task = Controller.TASK_PID_CONTROLLER
        #self.task = Controller.TASK_PD_FF_CONTROLLER
        self.task = Controller.TASK_BACKSTEPPING_CONTROLLER
        
        self.declare_parameter(
            'task',
            self.task,
            rcl_interfaces.msg.ParameterDescriptor(
                description="Task",
                type=rcl_interfaces.msg.ParameterType.PARAMETER_STRING,
                read_only=False,
                additional_constraints=f"Allowed values: {' '.join(Controller.TASKS)}"
            )
        )

        self.task = self.get_parameter("task").get_parameter_value().string_value
        self.p_gain = self.get_parameter("p_gain").get_parameter_value().double_array_value
        self.i_gain = self.get_parameter("i_gain").get_parameter_value().double_array_value
        self.d_gain = self.get_parameter("d_gain").get_parameter_value().double_array_value
        self.k1_gain = self.get_parameter("k1_gain").get_parameter_value().double_array_value
        self.k2_gain = self.get_parameter("k2_gain").get_parameter_value().double_array_value

        self.last_reference = None
        self.last_observation = None

        self.xi = np.zeros((3, 1), dtype=float)

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

        controller_period = 0.1
        self.controller_timer = self.create_timer(controller_period, self.controller_callback)

    def timer_callback(self):
        self.task = self.get_parameter("task").get_parameter_value().string_value
        self.p_gain = self.get_parameter("p_gain").get_parameter_value().double_array_value
        self.i_gain = self.get_parameter("i_gain").get_parameter_value().double_array_value
        self.d_gain = self.get_parameter("d_gain").get_parameter_value().double_array_value
        self.k1_gain = self.get_parameter("k1_gain").get_parameter_value().double_array_value
        self.k2_gain = self.get_parameter("k2_gain").get_parameter_value().double_array_value

        self.get_logger().info(f"Parameter task: {self.task}", throttle_duration_sec=1.0)

    def controller_callback(self):
        if self.last_reference is None or self.last_observation is None:
            self.get_logger().warn(
                "Last reference or last observation is None",
                throttle_duration_sec=1.0
            )
            return

        if Controller.TASK_PD_FF_CONTROLLER in self.task:
            tau = PD_FF_controller(
                self.last_observation,
                self.last_reference,
                self.p_gain,
                self.d_gain
            )

        elif Controller.TASK_PID_CONTROLLER in self.task:
            tau, self.xi = PID_controller(
                self.last_observation,
                self.last_reference,
                self.p_gain,
                self.i_gain,
                self.d_gain,
                self.xi,
                0.1
            )

        elif Controller.TASK_BACKSTEPPING_CONTROLLER in self.task:
            tau = backstepping_controller(
                self.last_observation,
                self.last_reference,
                self.k1_gain,
                self.k2_gain
            )

        else:
            tau = np.zeros((3, 1), dtype=float)

        tau = np.array(tau, dtype=float).reshape(-1)

        if tau.size != 3:
            self.get_logger().warn(
                f"tau has length {tau.size} but it should be 3: tau := [Fx, Fy, Mz]",
                throttle_duration_sec=1.0
            )
            return

        tau_cmd = std_msgs.msg.Float32MultiArray()
        tau_cmd.data = tau.tolist()
        self.pubs["tau_cmd"].publish(tau_cmd)

    def received_reference(self, msg):
        self.last_reference = msg

    def received_observer(self, msg):
        self.last_observation = msg


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(Controller())
    rclpy.shutdown()


if __name__ == '__main__':
    main()