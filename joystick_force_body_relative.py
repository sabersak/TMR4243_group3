#!/usr/bin/env python3

import sensor_msgs.msg
import numpy as np
from template_joystick_control.joystick_mapping import JoystickMapping
from Case_A import ThrusAllocator

allocator = ThrusAllocator()
allocator.setup_allocation_matrix()

def deadzone(x: float, dz: float = 0.05) -> float:
    return 0.0 if abs(x) < dz else x


def joystick_force_body_relative(
        joystick: sensor_msgs.msg.Joy,
        mapping: JoystickMapping) -> np.ndarray:
    # Replace the following line
    u0, u1, u2, a1, a2 = 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Typical convention: pushing stick forward gives negative axis value
    JOY_X = -deadzone(joystick.axes[mapping.LEFT_STICK_VERTICAL])      # surge
    JOY_Y =  deadzone(joystick.axes[mapping.LEFT_STICK_HORIZONTAL])    # sway

    # Choose yaw source (common: right stick horizontal)
    JOY_N =  deadzone(joystick.axes[mapping.RIGHT_STICK_HORIZONTAL])   # yaw

    # Simple gains (keep 1.0 unless you want scaling)
    kX, kY, kN = 1.0, 1.0, 1.0
    tau_cmd = np.array([kX * JOY_X, kY * JOY_Y, kN * JOY_N], dtype=float)

    # Task 2.5 allocator (varying azimuth)
    F_cmd, alpha_cmd, u_cmd, f_star, tau_err = allocator.allocate_task2(tau_cmd)

    u1, u2, u3 = u_cmd
    a1, a2 = alpha_cmd


    #
    ## Write your code below
    #
    u = np.array([[u0, u1, u2, a1, a2]], dtype=float).T
    return u