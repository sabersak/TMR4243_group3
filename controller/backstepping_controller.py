import numpy as np
from PD_FF_controller import body_pose_error, Rz

S = np.array([[0.0, -1.0, 0.0],
              [1.0,  0.0, 0.0],
              [0.0,  0.0, 0.0]], dtype=float)


def backstepping_controller(observation, reference, K1_diag, K2_diag):
    """
    Case C straight-line guidance, using:
      z1 = R^T(eta - eta_d(s))
      alpha1 = -K1 z1 + R^T eta_ds * v_s
      z2 = nu - alpha1
      tau = D nu + M alpha1_dot - K2 z2 - b_hat
    """
    # Vessel matrices
    M = np.array([[9.9, 0.0, 0.0],
                  [0.0, 10.8, 0.23],
                  [0.0, 0.23, 0.89]], dtype=float)
    D = np.array([[1.35, 0.0, 0.0],
                  [0.0, 14.25, 2.13],
                  [0.0, 2.13, 1.07]], dtype=float)

    eta = np.array(observation.eta, dtype=float).reshape(3)
    nu  = np.array(observation.nu, dtype=float).reshape(3)
    b_hat = np.array(observation.bias, dtype=float).reshape(3)

    eta_d  = np.array(reference.eta_d, dtype=float).reshape(3)
    eta_ds = np.array(reference.eta_ds, dtype=float).reshape(3)
    eta_ds2 = np.array(reference.eta_ds2, dtype=float).reshape(3)

    v_s = float(getattr(reference, 'v_s', 0.0))
    w   = float(getattr(reference, 'w', 0.0))
    v_ss = float(getattr(reference, 'v_ss', 0.0))  # interpreted as υ_s^s(s)

    s_dot = v_s + w

    K1 = np.diag(np.array(K1_diag, dtype=float).reshape(3))
    K2 = np.diag(np.array(K2_diag, dtype=float).reshape(3))

    psi = float(eta[2])
    r = float(nu[2])
    R = Rz(psi)

    # Step 1
    z1 = body_pose_error(eta, eta_d)
    q = eta_ds * v_s
    alpha1 = -K1 @ z1 + (R.T @ q)

    # z1_dot = -r S z1 + nu - R^T eta_ds * s_dot
    z1_dot = -r * (S @ z1) + nu - (R.T @ (eta_ds * s_dot))

    # q_dot = d/dt(eta_ds * v_s) = s_dot*(eta_ds2 * v_s + eta_ds * v_ss)
    q_dot = s_dot * (eta_ds2 * v_s + eta_ds * v_ss)

    # alpha1_dot = -K1 z1_dot + (R^T)_dot q + R^T q_dot
    # (R^T)_dot = -r S R^T
    alpha1_dot = -K1 @ z1_dot - r * (S @ (R.T @ q)) + (R.T @ q_dot)

    # Step 2
    z2 = nu - alpha1
    tau = (D @ nu) + (M @ alpha1_dot) - (K2 @ z2) - b_hat

    return tau.reshape(3, 1)