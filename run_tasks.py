import numpy as np
from thrust_allocator import ThrustAllocator
from joystick_controller import JoystickController


class TaskRunner:
    """All printing/demos for Case A report screenshots live here."""

    def __init__(self, allocator: ThrustAllocator, joystick: JoystickController):
        self.allocator = allocator
        self.joystick = joystick

    @staticmethod
    def header(title: str):
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)

    @staticmethod
    def print_results(F_cmd, u_cmd, alpha_cmd, tau_err):
        print("F_cmd     =", np.array2string(F_cmd, precision=4, floatmode="fixed", suppress_small=True))
        print("u_cmd     =", np.array2string(u_cmd, precision=4, floatmode="fixed", suppress_small=True))
        print("alpha_cmd =", np.array2string(alpha_cmd, precision=4, floatmode="fixed", suppress_small=True))
        print("tau_err   =", np.array2string(tau_err, precision=4, floatmode="fixed", suppress_small=True))

    def run_task_1_7(self):
        self.header("Task 1.7 (Fixed-azimuth allocation) — Case (a)")
        tau_cmd = np.array([1.0, -1.0, -0.5], dtype=float)
        alpha_fixed = np.array([np.pi/2, np.pi/2], dtype=float)
        F_cmd, u_cmd, alpha_cmd, F_star, tau_err = self.allocator.allocate_task1(tau_cmd, alpha_fixed)
        print("tau_cmd   =", tau_cmd, ", alpha_fixed =", alpha_fixed)
        print("F_star    =", np.array2string(F_star, precision=4, floatmode="fixed"))
        self.print_results(F_cmd, u_cmd, alpha_cmd, tau_err)

        self.header("Task 1.7 (Fixed-azimuth allocation) — Case (b)")
        tau_cmd = np.array([-2.0, 0.0, 0.0], dtype=float)
        alpha_fixed = np.array([np.pi, np.pi], dtype=float)
        F_cmd, u_cmd, alpha_cmd, F_star, tau_err = self.allocator.allocate_task1(tau_cmd, alpha_fixed)
        print("tau_cmd   =", tau_cmd, ", alpha_fixed =", alpha_fixed)
        print("F_star    =", np.array2string(F_star, precision=4, floatmode="fixed", suppress_small=True))
        self.print_results(F_cmd, u_cmd, alpha_cmd, tau_err)

    def run_task_2_5(self):
        self.header("Task 2.5 (Varying-azimuth allocation) — Case (a)")
        tau_cmd = np.array([1.0, -1.0, -0.5], dtype=float)
        F_cmd, alpha_cmd, u_cmd, f_star, tau_err = self.allocator.allocate_task2(tau_cmd)
        print("tau_cmd   =", tau_cmd)
        print("f_star    =", np.array2string(f_star, precision=4, floatmode="fixed", suppress_small=True))
        self.print_results(F_cmd, u_cmd, alpha_cmd, tau_err)

        self.header("Task 2.5 (Varying-azimuth allocation) — Case (b)")
        tau_cmd = np.array([-2.0, 0.0, 0.0], dtype=float)
        F_cmd, alpha_cmd, u_cmd, f_star, tau_err = self.allocator.allocate_task2(tau_cmd)
        print("tau_cmd   =", tau_cmd)
        print("f_star    =", np.array2string(f_star, precision=4, floatmode="fixed", suppress_small=True))
        self.print_results(F_cmd, u_cmd, alpha_cmd, tau_err)

    def run_task_3_3(self):
        # Case (a)
        JOYcmd_a = np.array([1.0, 0.0, 0.0], dtype=float)
        psi_rel_a = np.deg2rad(-90.0)

        self.header("Task 3.3 — Case (a): JOYcmd=[1,0,0], psi_rel=-90 deg")

        tau_cmd, F_cmd, u_cmd, alpha_cmd, tau_err = self.joystick.body_relative(JOYcmd_a)
        print("\n[Body-relative]")
        print("tau_cmd   =", np.array2string(tau_cmd, precision=4, floatmode="fixed", suppress_small=True))
        self.print_results(F_cmd, u_cmd, alpha_cmd, tau_err)

        tau_cmd, F_cmd, u_cmd, alpha_cmd, tau_err = self.joystick.basin_relative(JOYcmd_a, psi_rel_a)
        print("\n[Basin-relative]")
        print("tau_cmd   =", np.array2string(tau_cmd, precision=4, floatmode="fixed", suppress_small=True))
        self.print_results(F_cmd, u_cmd, alpha_cmd, tau_err)

        # Case (b)
        JOYcmd_b = np.array([0.0, 1.0, 0.0], dtype=float)
        psi_rel_b = np.deg2rad(180.0)

        self.header("Task 3.3 — Case (b): JOYcmd=[0,1,0], psi_rel=180 deg")

        tau_cmd, F_cmd, u_cmd, alpha_cmd, tau_err = self.joystick.body_relative(JOYcmd_b)
        print("\n[Body-relative]")
        print("tau_cmd   =", np.array2string(tau_cmd, precision=4, floatmode="fixed", suppress_small=True))
        self.print_results(F_cmd, u_cmd, alpha_cmd, tau_err)

        tau_cmd, F_cmd, u_cmd, alpha_cmd, tau_err = self.joystick.basin_relative(JOYcmd_b, psi_rel_b)
        print("\n[Basin-relative]")
        print("tau_cmd   =", np.array2string(tau_cmd, precision=4, floatmode="fixed", suppress_small=True))
        self.print_results(F_cmd, u_cmd, alpha_cmd, tau_err)



if __name__ == "__main__":
    allocator = ThrustAllocator()
    allocator.setup_allocation_matrix()

    joystick = JoystickController(allocator)
    runner = TaskRunner(allocator, joystick)

    runner.run_task_1_7()
    runner.run_task_2_5()
    runner.run_task_3_3()
