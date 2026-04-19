import numpy as np

def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0*np.pi) - np.pi

def yaw_error_pi(psi: float, psi_d: float) -> float:
    return wrap_pi(wrap_pi(psi) - wrap_pi(psi_d))

def Rz(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def body_pose_error(eta: np.ndarray, eta_d: np.ndarray) -> np.ndarray:
    """
    e = R(psi)^T (eta - eta_d)
    """
    psi = float(eta[2])
    dp = eta[0:2] - eta_d[0:2]
    dpsi = yaw_error_pi(float(eta[2]), float(eta_d[2]))
    d = np.array([dp[0], dp[1], dpsi], dtype=float)
    return Rz(psi).T @ d

def PD_FF_controller(observation, reference, Kp_diag, Kd_diag):
    """
    Case C Eq. (6): tau = -Kp R^T (eta_hat - eta_ref) - Kd nu_hat - b_hat
    """
    eta = np.array(observation.eta, dtype=float).reshape(3)
    nu = np.array(observation.nu, dtype=float).reshape(3)
    b_hat = np.array(observation.bias, dtype=float).reshape(3)

    eta_ref = np.array(reference.eta_d, dtype=float).reshape(3)

    Kp = np.diag(np.array(Kp_diag, dtype=float).reshape(3))
    Kd = np.diag(np.array(Kd_diag, dtype=float).reshape(3))

    e = body_pose_error(eta, eta_ref)
    tau = -Kp @ e - Kd @ nu - b_hat
    return tau.reshape(3, 1)