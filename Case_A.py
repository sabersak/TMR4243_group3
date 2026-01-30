import numpy as np

class ThrusAllocator:
    """
    Case A thrust allocation:
      - Task 1.7 (fixed azimuth): tau = B(alpha) F, with F = K u
      - Task 2.5 (varying azimuth):  tau = B_rect f, with f = [X1,Y1,X2,Y2,Y3]^T

    Defaults:
      K = I, W = I, F_d = 0 (Task 1)
      W = I, f_d = 0 (Task 2)
    """

    def __init__(self):
        # Geometry (meters)
        self.lx1, self.ly1 = -0.415, -0.070
        self.lx2, self.ly2 = -0.415, 0.070
        self.lx3 = 0.370

        # Task 2 constant allocation matrix
        self.B_rect = np.empty((3, 5), dtype=float)

        # Bounds
        self.u1_min, self.u1_max = 0.0, 1.0
        self.u2_min, self.u2_max = 0.0, 1.0
        self.u3_min, self.u3_max = -1.0, 1.0

        # Defaults (Chosen values for K,W, and F_d)
        self.K = np.eye(3, dtype=float)         # F = K u

        self.W1 = np.eye(3, dtype=float)         # Task 1 weights
        self.Fd1 = np.zeros(3, dtype=float)      # Task 1 preference

        self.W2 = np.eye(5, dtype=float)         # Task 2 weights
        self.fd2 = np.zeros(5, dtype=float)      # Task 2 preference


    # ---------------- Task 2 (varying azimuth) ----------------
    def setup_allocation_matrix(self):
        self.B_rect = np.array([
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1],
            [-self.ly1, self.lx1, -self.ly2, self.lx2, self.lx3]
        ], dtype=float)

    def enforce_limits_mu(self, f):
        """
        Validity μ for rectangular forces:
          - clip Y3 to [u3_min,u3_max]
          - scale (X1,Y1) and (X2,Y2) if magnitude exceeds 1
        """
        f_cmd = f.astype(float).copy()

        # Tunnel saturation
        f_cmd[4] = np.clip(f_cmd[4], self.u3_min, self.u3_max)

        # Thruster 1 scaling
        F1 = np.hypot(f_cmd[0], f_cmd[1])
        if F1 > self.u1_max and F1 > 1e-12:         # F1 > 1e-12 is to avoid division by 0
            s = self.u1_max / F1
            f_cmd[0] *= s
            f_cmd[1] *= s

        # Thruster 2 scaling
        F2 = np.hypot(f_cmd[2], f_cmd[3])
        if F2 > self.u2_max and F2 > 1e-12:
            s = self.u2_max / F2
            f_cmd[2] *= s
            f_cmd[3] *= s

        return f_cmd

    def allocate_task2(self, tau_cmd):
        """
        Task 2.5:
          f* = B_W^† tau_cmd + Q_W f_d
          then μ (limits), then map to (F_cmd, alpha_cmd)

        Returns:
          F_cmd, alpha_cmd, u_cmd, f_star, tau_err 
        """
        tau_cmd = np.asarray(tau_cmd, dtype=float).reshape(3)
        W = self.W2
        f_d = self.fd2

        Bwdag = weighted_pinv(self.B_rect, W)
        Qw = np.eye(5) - Bwdag @ self.B_rect

        f_star = Bwdag @ tau_cmd + Qw @ f_d
        f_cmd = self.enforce_limits_mu(f_star)

        # Map to polar
        X1, Y1, X2, Y2, Y3 = f_cmd
        F1 = np.hypot(X1, Y1)
        F2 = np.hypot(X2, Y2)
        F3 = Y3

        alpha1 = wrap_pi(np.arctan2(Y1, X1))
        alpha2 = wrap_pi(np.arctan2(Y2, X2))

        F_cmd = np.array([F1, F2, F3], dtype=float)
        alpha_cmd = np.array([alpha1, alpha2], dtype=float)
        u_cmd =  np.linalg.solve(self.K, F_cmd)  # u* = K^{-1} F*

        # Calculating tau_error = applied - command
        tau_applied = self.B_rect @ f_cmd
        tau_err = tau_cmd - tau_applied

        return F_cmd, alpha_cmd, u_cmd, f_star, tau_err

    # ---------------- Task 1 (fixed azimuth) ----------------
    def B_fixed(self, alpha1, alpha2):
        c1, s1 = np.cos(alpha1), np.sin(alpha1)
        c2, s2 = np.cos(alpha2), np.sin(alpha2)
        return np.array([
            [c1,                           c2,                           0.0],
            [s1,                           s2,                           1.0],
            [self.lx1 * s1 - self.ly1 * c1, self.lx2 * s2 - self.ly2 * c2, self.lx3]
        ], dtype=float)

    def enforce_limits_gamma(self, u_star, alpha_fixed):
        """
        bounds are on u + If u_i < 0 for azimuth thrusters, flip sign and add pi to alpha_i.
        """
        u = u_star.astype(float).copy()
        a = alpha_fixed.astype(float).copy()

        for i in (0, 1):
            if u[i] < 0:
                u[i] = -u[i]
                a[i] = wrap_pi(a[i] + np.pi)

        u[0] = np.clip(u[0], self.u1_min, self.u1_max)
        u[1] = np.clip(u[1], self.u2_min, self.u2_max)
        u[2] = np.clip(u[2], self.u3_min, self.u3_max)

        a[0] = wrap_pi(a[0])
        a[1] = wrap_pi(a[1])

        return u, a

    def allocate_task1(self, tau_cmd, alpha_fixed):
        """
        Task 1.7:
          F* = B_W^† tau_cmd + Q_W F_d
          u* = K^{-1} F*
          then enforce limits, then F_cmd = K u_cmd

        Returns:
          F_cmd, u_cmd, alpha_cmd, F_star, tau_err
        """
        tau_cmd = np.asarray(tau_cmd, dtype=float).reshape(3)
        alpha_fixed = np.asarray(alpha_fixed, dtype=float).reshape(2)

        K = self.K
        W = self.W1
        F_d = self.Fd1 

        B = self.B_fixed(alpha_fixed[0], alpha_fixed[1])

        Bwdag = weighted_pinv(B, W)
        Qw = np.eye(3) - Bwdag @ B

        F_star = Bwdag @ tau_cmd + Qw @ F_d
        u_star = np.linalg.solve(K, F_star)  # u* = K^{-1} F*

        u_cmd, alpha_cmd = self.enforce_limits_gamma(u_star, alpha_fixed)

        F_cmd = K @ u_cmd
        tau_applied = B @ F_cmd
        tau_err = tau_cmd - tau_applied


        return F_cmd, u_cmd, alpha_cmd, F_star, tau_err


# ---------------- utilities ----------------
def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def weighted_pinv(B, W):
    """ B_W^† = W^{-1} B^T (B W^{-1} B^T)^-1 """
    W_inv = np.linalg.inv(W)
    return W_inv @ B.T @ np.linalg.pinv(B @ W_inv @ B.T)

def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

# ----------------Printing results ----------------
def print_case_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def print_results(F_cmd, u_cmd, alpha_cmd, tau_err):
    print("F_cmd     =", np.array2string(F_cmd, precision=4, floatmode="fixed", suppress_small=True))
    print("u_cmd     =", np.array2string(u_cmd, precision=4, floatmode="fixed", suppress_small=True))
    print("alpha_cmd =", np.array2string(alpha_cmd, precision=4, floatmode="fixed", suppress_small=True))
    print("tau_err   =", np.array2string(tau_err, precision=4, floatmode="fixed", suppress_small=True))


# ---------------- Task 3 functions (same style as your templates) ----------------
def joystick_force_body_relative(
        JOYcmd: np.ndarray,
        allocator: ThrusAllocator) -> np.ndarray:
    """
    Task 3.1: body-relative: tau_cmd(body) = JOYcmd
    """
    tau_cmd = np.asarray(JOYcmd, dtype=float).reshape(3)

    F_cmd, alpha_cmd, u_cmd, f_star, tau_err = allocator.allocate_task2(tau_cmd)
    return tau_cmd, F_cmd, u_cmd, alpha_cmd

def joystick_force_basin_relative(
        JOYcmd: np.ndarray,
        psi_rel: float,
        allocator: ThrusAllocator) -> np.ndarray:
    """
    Task 3.2: basin-relative: tau_cmd(body) = R(psi_rel)^T * JOYcmd
    """
    JOYcmd = np.asarray(JOYcmd, dtype=float).reshape(3)

    tau_cmd = Rz(psi_rel).T @ JOYcmd

    F_cmd, alpha_cmd, u_cmd, f_star, tau_err = allocator.allocate_task2(tau_cmd)
    return tau_cmd, F_cmd, u_cmd, alpha_cmd


# ---------------- printing for Task 3.3 ----------------

def print_results_task3(tag, tau_cmd, F_cmd, u_cmd, alpha_cmd):
    print(f"\n[{tag}]")
    print("tau_cmd   =", np.array2string(tau_cmd, precision=4, floatmode="fixed", suppress_small=True))
    print("F_cmd     =", np.array2string(F_cmd, precision=4, floatmode="fixed", suppress_small=True))
    print("u_cmd     =", np.array2string(u_cmd, precision=4, floatmode="fixed", suppress_small=True))
    print("alpha_cmd =", np.array2string(alpha_cmd, precision=4, floatmode="fixed", suppress_small=True))

if __name__ == "__main__":
    allocator = ThrusAllocator()
    allocator.setup_allocation_matrix()

    # ---------------- Task 1.7 cases ----------------
    print_case_header("Task 1.7 (Fixed-azimuth allocation) — Case (a)")
    tau_cmd = np.array([1.0, -1.0, -0.5], dtype=float)
    alpha_fixed = np.array([np.pi / 2, np.pi / 2], dtype=float)
    F_cmd, u_cmd, alpha_cmd, F_star, tau_err = allocator.allocate_task1(tau_cmd, alpha_fixed)
    print("tau_cmd   =", tau_cmd, ", alpha_fixed =", alpha_fixed)
    print("F_star    =", np.array2string(F_star, precision=4, floatmode="fixed"))
    print_results(F_cmd, u_cmd, alpha_cmd, tau_err)

    print_case_header("Task 1.7 (Fixed-azimuth allocation) — Case (b)")
    tau_cmd = np.array([-2.0, 0.0, 0.0], dtype=float)
    alpha_fixed = np.array([np.pi, np.pi], dtype=float)
    F_cmd, u_cmd, alpha_cmd, F_star, tau_err = allocator.allocate_task1(tau_cmd, alpha_fixed)
    print("tau_cmd   =", tau_cmd, ", alpha_fixed =", alpha_fixed)
    print("F_star    =", np.array2string(F_star, precision=4, floatmode="fixed", suppress_small=True))
    print_results(F_cmd, u_cmd, alpha_cmd, tau_err)

    # ---------------- Task 2.5 cases ----------------
    print_case_header("Task 2.5 (Varying-azimuth allocation) — Case (a)")
    tau_cmd = np.array([1.0, -1.0, -0.5], dtype=float)
    F_cmd, alpha_cmd, u_cmd, f_star, tau_err = allocator.allocate_task2(tau_cmd)
    print("tau_cmd   =", tau_cmd)
    print("f_star    =", np.array2string(f_star, precision=4, floatmode="fixed", suppress_small=True))
    print_results(F_cmd, u_cmd, alpha_cmd, tau_err)

    print_case_header("Task 2.5 (Varying-azimuth allocation) — Case (b)")
    tau_cmd = np.array([-2.0, 0.0, 0.0], dtype=float)
    F_cmd, alpha_cmd, u_cmd, f_star, tau_err = allocator.allocate_task2(tau_cmd)
    print("tau_cmd   =", tau_cmd)
    print("f_star    =", np.array2string(f_star, precision=4, floatmode="fixed", suppress_small=True))
    print_results(F_cmd, u_cmd, alpha_cmd, tau_err)



    # ---------------- Task 3.3 cases ----------------

    # --- Case (a): JOYcmd=[1,0,0], psi-psi_mc = -90 deg ---
    JOYcmd_a = np.array([1.0, 0.0, 0.0], dtype=float)
    psi_rel_a = np.deg2rad(-90.0)

    print_case_header("Task 3.3 — Case (a): JOYcmd=[1,0,0], psi_rel=-90 deg")

    tau_cmd, F_cmd, u_cmd, alpha_cmd = joystick_force_body_relative(JOYcmd_a, allocator)
    print_results_task3("Body-relative", tau_cmd, F_cmd, u_cmd, alpha_cmd)

    tau_cmd, F_cmd, u_cmd, alpha_cmd = joystick_force_basin_relative(JOYcmd_a, psi_rel_a, allocator)
    print_results_task3("Basin-relative", tau_cmd, F_cmd, u_cmd, alpha_cmd)

    # --- Case (b): JOYcmd=[0,1,0], psi-psi_mc = 180 deg ---
    JOYcmd_b = np.array([0.0, 1.0, 0.0], dtype=float)
    psi_rel_b = np.deg2rad(180.0)

    print_case_header("Task 3.3 — Case (b): JOYcmd=[0,1,0], psi_rel=180 deg")

    tau_cmd, F_cmd, u_cmd, alpha_cmd = joystick_force_body_relative(JOYcmd_b, allocator)
    print_results_task3("Body-relative", tau_cmd, F_cmd, u_cmd, alpha_cmd)

    tau_cmd, F_cmd, u_cmd, alpha_cmd = joystick_force_basin_relative(JOYcmd_b, psi_rel_b, allocator)
    print_results_task3("Basin-relative", tau_cmd, F_cmd, u_cmd, alpha_cmd)