from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def trigger_to_01(axis_value: float) -> float:
    """
    DS4: released=+1, pressed=-1
    Map to: released=0, pressed=1
    """
    return 0.5 * (1.0 - float(axis_value))

@dataclass
class StraightLinePath:
    p0: np.ndarray  # shape (2,)
    p1: np.ndarray  # shape (2,)

    @property
    def d(self) -> np.ndarray:
        return self.p1 - self.p0

    @property
    def d_norm(self) -> float:
        return float(np.linalg.norm(self.d))

    def pd(self, s: float) -> np.ndarray:
        return self.p0 + s * self.d

    def eta_d(self, s: float, psi_ref: float) -> np.ndarray:
        p = self.pd(s)
        return np.array([p[0], p[1], psi_ref], dtype=float)

    def eta_ds(self) -> np.ndarray:
        # eta_d^s = col(p1-p0, 0)
        d = self.d
        return np.array([d[0], d[1], 0.0], dtype=float)

    def eta_ds2(self) -> np.ndarray:
        # straight line => second derivative is zero
        return np.array([0.0, 0.0, 0.0], dtype=float)

def v_s_constant(Uref: float, path: StraightLinePath) -> float:
    """
    Speed assignment: |p_dot| = Uref => s_dot = Uref / ||p_d^s||
    """
    if path.d_norm <= 1e-9:
        return 0.0
    return float(Uref) / path.d_norm

def V1_s_for_line(p: np.ndarray, s: float, path: StraightLinePath) -> float:
    """
    For straight line + constant heading:
      V1^s(eta,s) = - (p1-p0)^T (p - p_d(s))
    """
    d = path.d
    return -float(d.T @ (p - path.pd(s)))

class StraightLineGuidance:
    """
    Maintains s(t) and (optionally) omega state for filtered update law.
    Produces eta_d(s), eta_d^s, eta_d^{s2} and (w, v_s, v_ss) where:
      s_dot = v_s + w
      v_ss is υ_s^s(s) (derivative wrt s), so v_dot = v_ss * s_dot
    """
    def __init__(self, p0: np.ndarray, p1: np.ndarray):
        self.path = StraightLinePath(p0=np.array(p0, dtype=float).reshape(2),
                                     p1=np.array(p1, dtype=float).reshape(2))
        self.s = 0.0
        self.omega = 0.0

    def reset(self, s0: float = 0.0):
        self.s = float(s0)
        self.omega = 0.0

    def update(self,
               p_hat: np.ndarray,
               psi_ref: float,
               Uref: float,
               mu: float,
               eps: float,
               lam: float,
               update_mode: str,
               dt: float,
               clamp_segment: bool,
               sigma: float):
        """
        Args:
          p_hat: estimated position (2,)
          sigma: activation in [0,1]. If sigma=0 => s_dot = 0.

        Returns:
          eta_d, eta_ds, eta_ds2  as (3,1)
          w, v_s, v_ss            as floats
          s                       updated s
        """
        dt = max(float(dt), 1e-6)
        sigma = float(np.clip(sigma, 0.0, 1.0))

        # If not in maneuvering mode => freeze s(t) and publish "no motion"
        if sigma <= 1e-9:
            eta_d = self.path.eta_d(self.s, psi_ref).reshape(3, 1)
            eta_ds = self.path.eta_ds().reshape(3, 1)
            eta_ds2 = self.path.eta_ds2().reshape(3, 1)
            return eta_d, eta_ds, eta_ds2, 0.0, 0.0, 0.0, float(self.s)

        # Base speed assignment υ_s(s) (constant for straight line)
        v_s_base = v_s_constant(Uref, self.path)
        v_ss = 0.0  # υ_s^s(s) = 0 for constant speed assignment

        # Update law term
        p_hat = np.array(p_hat, dtype=float).reshape(2)
        V1s = V1_s_for_line(p_hat, self.s, self.path)
        mode = str(update_mode).lower()

        if mode == "tracking":
            omega_cmd = 0.0
        elif mode == "gradient":
            omega_cmd = -mu * V1s
        elif mode == "normalized":
            omega_cmd = -(mu / (self.path.d_norm + eps)) * V1s
        elif mode == "filtered":
            # omega_dot = -lam (omega + mu V1s)
            self.omega += dt * (-lam * (self.omega + mu * V1s))
            omega_cmd = self.omega
        else:
            omega_cmd = 0.0

        s_dot_raw = sigma * (v_s_base + omega_cmd)

        # Integrate and (optional) clamp s
        s_prev = float(self.s)
        s_next = s_prev + dt * s_dot_raw
        if clamp_segment:
            s_next = float(np.clip(s_next, 0.0, 1.0))

        # Effective s_dot consistent with saturation
        s_dot_eff = (s_next - s_prev) / dt
        self.s = s_next

        # Decompose s_dot = v_s + w consistently
        # If stuck at endpoint: stop all progression (v_s=w=0)
        if clamp_segment and (
            (self.s <= 0.0 + 1e-9 and s_dot_eff <= 0.0) or
            (self.s >= 1.0 - 1e-9 and s_dot_eff >= 0.0)
        ):
            v_s_eff = 0.0
            w_eff = 0.0
        else:
            v_s_eff = sigma * v_s_base
            w_eff = s_dot_eff - v_s_eff

        eta_d = self.path.eta_d(self.s, psi_ref).reshape(3, 1)
        eta_ds = self.path.eta_ds().reshape(3, 1)
        eta_ds2 = self.path.eta_ds2().reshape(3, 1)

        return eta_d, eta_ds, eta_ds2, float(w_eff), float(v_s_eff), float(v_ss), float(self.s)