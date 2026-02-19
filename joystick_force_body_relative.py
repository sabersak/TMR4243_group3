#!/usr/bin/env python3

import sensor_msgs.msg
import numpy as np
from joystick_mapping import JoystickMapping
from thrust_allocator import ThrustAllocator

allocator = ThrustAllocator()
allocator.setup_allocation_matrix()

def deadzone(x: float, dz: float = 0.05) -> float:
    return 0.0 if abs(x) < dz else x

def trigger_to_01(axis_value: float) -> float:
    """
    DS4: released=+1, pressed=-1
    Map to: released=0, pressed=1
    """
    return 0.5 * (1.0 - float(axis_value))

def joystick_force_body_relative(
        joystick: sensor_msgs.msg.Joy,
        mapping: JoystickMapping,
        allocation_mode: str = "varying") -> np.ndarray:

    JOY_X = deadzone(joystick.axes[mapping.LEFT_STICK_VERTICAL])     # surge
    JOY_Y = -deadzone(joystick.axes[mapping.LEFT_STICK_HORIZONTAL])   # sway

    l2 = trigger_to_01(joystick.axes[mapping.LEFT_TRIGGER])
    r2 = trigger_to_01(joystick.axes[mapping.RIGHT_TRIGGER])
    JOY_N = r2 - l2                                                   # yaw moment

    tau_cmd = np.array([JOY_X, JOY_Y, JOY_N], dtype=float)

    allocation_mode = (allocation_mode or "varying").lower()

    if allocation_mode == "fixed":
        alpha_fixed = np.array([0.0, 0.0], dtype=float)
        F_cmd, u_cmd3, alpha_cmd, F_star, tau_err = allocator.allocate_task1(tau_cmd, alpha_fixed)
        u1, u2, u3 = u_cmd3
        a1, a2 = alpha_fixed
    else:
        F_cmd, alpha_cmd, u_cmd3, f_star, tau_err = allocator.allocate_task2(tau_cmd)
        u1, u2, u3 = u_cmd3
        a1, a2 = alpha_cmd

    

    # Convert (u1,u2,u3) -> [u0,u1,u2,a1,a2]
    u = np.array([[float(u3), float(u1), float(u2), float(a1), float(a2)]], dtype=float).T
    return u