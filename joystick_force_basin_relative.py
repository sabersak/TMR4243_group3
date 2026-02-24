#!/usr/bin/env python3
import sensor_msgs.msg
import numpy as np
from joystick_mapping import JoystickMapping
from thrust_allocator import ThrustAllocator
from joystick_force_body_relative import deadzone, trigger_to_01

def joystick_force_basin_relative(
        joystick: sensor_msgs.msg.Joy,
        position: np.ndarray,
        mapping: JoystickMapping,
        psi_mc: float = 0.0) -> np.ndarray:
    """
    Joystick -> tau_cmd (basin) -> tau_cmd (body)
    Returns tau_cmd(body) = Rz(psi_rel)^T * tau_basin as (3,1)
    """
    eta = np.asarray(position, dtype=float).reshape(3)
    psi_rel = float(eta[2]) - float(psi_mc)

    # joystick interpreted in basin frame
    JOY_X = deadzone(joystick.axes[mapping.LEFT_STICK_VERTICAL])
    JOY_Y = -deadzone(joystick.axes[mapping.LEFT_STICK_HORIZONTAL])

    l2 = trigger_to_01(joystick.axes[mapping.LEFT_TRIGGER])
    r2 = trigger_to_01(joystick.axes[mapping.RIGHT_TRIGGER])
    JOY_N = r2 - l2

    tau_basin = np.array([JOY_X, JOY_Y, JOY_N], dtype=float)

    # rotate basin -> body
    tau_body = ThrustAllocator.Rz(psi_rel).T @ tau_basin
    return np.asarray(tau_body, dtype=float).reshape(3, 1)