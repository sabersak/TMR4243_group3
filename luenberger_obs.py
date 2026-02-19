import numpy as np

def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class LuenbergerObserver:
    """
    Luenberger observer with bias estimation.
    Discrete-time Euler step.
    """

    def __init__(self):
        self.M = np.array([[9.9, 0.0, 0.0], 
                           [0.0, 10.8, 0.23],
                           [0.0, 0.23, 0.89]], dtype=float)
        
        self.D = np.array([[1.35, 0.0, 0.0],
                           [0.0, 14.25, 2.13],
                           [0.0, 2.13, 1.07]], dtype=float)
        
        self.M_inv = np.linalg.inv(self.M)

        self.Tb = 142.0 * np.eye(3)
        self.Tb_inv = np.linalg.inv(self.Tb)

        self.eta_hat = np.zeros((3,1))
        self.nu_hat  = np.zeros((3,1))
        self.b_hat   = np.zeros((3,1))

    def step(self, eta_meas, tau_meas, L1, L2, L3, dt, dead_reckoning=False):
        eta = np.asarray(eta_meas).reshape(3,1)
        tau = np.asarray(tau_meas).reshape(3,1)

        psi = wrap_pi(float(eta[2,0]))
        R = Rz(psi)

        y_tilde = eta - self.eta_hat
        y_tilde[2,0] = wrap_pi(y_tilde[2,0])  # yaw wrapping

        if dead_reckoning:
            y_tilde[:] = 0.0

        # Eq. (11)
        eta_dot = R @ self.nu_hat + L1 @ y_tilde
        nu_dot  = self.M_inv @ (-self.D @ self.nu_hat + self.b_hat + tau + L2 @ (R.T @ y_tilde))
        b_dot   = -self.Tb_inv @ self.b_hat + L3 @ (R.T @ y_tilde)

        # Euler integration
        self.eta_hat = self.eta_hat + dt * eta_dot
        self.nu_hat  = self.nu_hat  + dt * nu_dot
        self.b_hat   = self.b_hat   + dt * b_dot

        self.eta_hat[2,0] = wrap_pi(float(self.eta_hat[2,0]))

        return self.eta_hat, self.nu_hat, self.b_hat
