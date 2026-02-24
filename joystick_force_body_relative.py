#!/usr/bin/env python3
import sensor_msgs.msg
import numpy as np
from joystick_mapping import JoystickMapping

def deadzone(x: float, dz: float = 0.05) -> float:
    return 0.0 if abs(x) < dz else x

def trigger_to_01(axis_value: float) -> float:
    """DS4: released=+1, pressed=-1 -> released=0, pressed=1"""
    return 0.5 * (1.0 - float(axis_value))

def joystick_force_body_relative(
        joystick: sensor_msgs.msg.Joy,
        mapping: JoystickMapping) -> np.ndarray:
    """
    Joystick -> tau_cmd (body)
    Returns tau_cmd = [X, Y, N]^T as (3,1)
    """
    JOY_X = deadzone(joystick.axes[mapping.LEFT_STICK_VERTICAL])        # surge
    JOY_Y = -deadzone(joystick.axes[mapping.LEFT_STICK_HORIZONTAL])     # sway

    l2 = trigger_to_01(joystick.axes[mapping.LEFT_TRIGGER])
    r2 = trigger_to_01(joystick.axes[mapping.RIGHT_TRIGGER])
    JOY_N = r2 - l2                                                     # yaw moment

    tau_cmd = np.array([JOY_X, JOY_Y, JOY_N], dtype=float).reshape(3, 1)
    return tau_cmd