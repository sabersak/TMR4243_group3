import numpy as np
from PD_FF_controller import body_pose_error


def PID_controller(observation, reference, Kp_diag, Ki_diag, Kd_diag, xi: np.ndarray, dt: float):
    """
    Case C Eq. (7):
      xi_dot = e
      tau = -Ki xi - Kp e - Kd nu
    """
    eta = np.array(observation.eta, dtype=float).reshape(3)
    nu = np.array(observation.nu, dtype=float).reshape(3)
    eta_ref = np.array(reference.eta_d, dtype=float).reshape(3)

    Kp = np.diag(np.array(Kp_diag, dtype=float).reshape(3))
    Ki = np.diag(np.array(Ki_diag, dtype=float).reshape(3))
    Kd = np.diag(np.array(Kd_diag, dtype=float).reshape(3))

    e = body_pose_error(eta, eta_ref)  
    xi = np.array(xi, dtype=float).reshape(3)
    xi = xi + float(dt) * e

    tau = -Ki @ xi - Kp @ e - Kd @ nu
    return tau.reshape(3, 1), xi.reshape(3,)