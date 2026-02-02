import numpy as np
from thrust_allocator import ThrustAllocator


class JoystickController:
    """
    Task 3 joystick mappings:
      - body-relative: tau_cmd = JOYcmd
      - basin-relative: tau_cmd = R(psi_rel)^T JOYcmd
    Then uses allocator.allocate_task2(tau_cmd) to compute (F_cmd, u_cmd, alpha_cmd).
    """

    def __init__(self, allocator: ThrustAllocator):
        self.allocator = allocator

    def body_relative(self, JOYcmd):
        """
        Task 3.1: tau_cmd(body) = JOYcmd
        """
        tau_cmd = np.asarray(JOYcmd, dtype=float).reshape(3)
        F_cmd, alpha_cmd, u_cmd, f_star, tau_err = self.allocator.allocate_task2(tau_cmd)
        return tau_cmd, F_cmd, u_cmd, alpha_cmd, tau_err

    def basin_relative(self, JOYcmd, psi_rel: float):
        """
        Task 3.2: tau_cmd(body) = R(psi_rel)^T * JOYcmd
        """
        JOYcmd = np.asarray(JOYcmd, dtype=float).reshape(3)
        tau_cmd = ThrustAllocator.Rz(psi_rel).T @ JOYcmd
        F_cmd, alpha_cmd, u_cmd, f_star, tau_err = self.allocator.allocate_task2(tau_cmd)
        return tau_cmd, F_cmd, u_cmd, alpha_cmd, tau_err
