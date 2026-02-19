#!/usr/bin/env python3

import sensor_msgs.msg
import numpy as np
from joystick_mapping import JoystickMapping
from thrust_allocator import ThrustAllocator
from joystick_force_body_relative import deadzone, trigger_to_01

allocator = ThrustAllocator()
allocator.setup_allocation_matrix()

PSI_MC = 0.0  # initial heading of the basin frame


def basin_to_body(tau_basin: np.ndarray, psi: float) -> np.ndarray:
    psi_rel = float(psi) - float(PSI_MC)
    return ThrustAllocator.Rz(psi_rel).T @ tau_basin


def joystick_force_basin_relative(
        joystick: sensor_msgs.msg.Joy,
        position: np.ndarray,
        mapping: JoystickMapping,
        allocation_mode: str = "varying",
        psi_mc: float = 0.0) -> np.ndarray:
    
    eta = np.asarray(position, dtype=float).reshape(3)
    psi = float(eta[2])
    psi_rel = psi - float(psi_mc)

    JOY_X = deadzone(joystick.axes[mapping.LEFT_STICK_VERTICAL])     # basin surge
    JOY_Y = -deadzone(joystick.axes[mapping.LEFT_STICK_HORIZONTAL])   # basin sway
    l2 = trigger_to_01(joystick.axes[mapping.LEFT_TRIGGER])
    r2 = trigger_to_01(joystick.axes[mapping.RIGHT_TRIGGER])
    JOY_N = r2 - l2

    tau_basin = np.array([JOY_X, JOY_Y, JOY_N], dtype=float)

    # Rotate basin to body 
    tau_body = ThrustAllocator.Rz(psi_rel).T @ tau_basin

    allocation_mode = (allocation_mode or "varying").lower()

    if allocation_mode == "fixed":
        alpha_fixed = np.array([0.0, 0.0], dtype=float)
        F_cmd, u_cmd3, alpha_cmd, F_star, tau_err = allocator.allocate_task1(tau_body, alpha_fixed)
        u1, u2, u3 = u_cmd3
        a1, a2 = alpha_fixed
    else:
        F_cmd, alpha_cmd, u_cmd3, f_star, tau_err = allocator.allocate_task2(tau_body)
        u1, u2, u3 = u_cmd3
        a1, a2 = alpha_cmd

    #tau = np.array([[float(tau0), float(tau1), float(tau2), float(a1), float(a2)]], dtype=float).T
    #return tau
    u = np.array([[float(u3), float(u1), float(u2), float(a1), float(a2)]], dtype=float).T
    return u