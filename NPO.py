import numpy as np
import matplotlib.pyplot as plt

class NPOObserver:

    def __init__(self):

        self.M  = np.array([
        [9.9,  0.0,  0.0],
        [0.0, 10.8,  0.23],
        [0.0,  0.23, 0.89]
        ])
        self.D  = np.array([
        [1.35,  0.0,  0.0],
        [0.0, 14.25, 2.13],
        [0.0,  2.13, 1.07]
        ])
        self.Ab = 0
        self.omega_c = 0.2
        self.Tb = 142.0 * np.eye(3)
        self.L1 = self.omega_c * np.eye(3)
        self.L2 = 0.5 * self.omega_c * np.eye(3)
        self.L3 = 0.1 * self.omega_c * np.eye(3)

        self.M_inv = np.linalg.inv(self.M)
        self.Tb_inv = np.linalg.inv(self.Tb)

        self.eta_hat = np.zeros(3)
        self.nu_hat  = np.zeros(3)
        self.b_hat   = np.zeros(3)


    def wrap_angle(self, rad: float) -> float:
        return (rad + np.pi) % (2.0 * np.pi) - np.pi
    

    def R(self, psi: float) -> np.ndarray:
        c = np.cos(psi)
        s = np.sin(psi)
        return np.array([
            [ c, -s, 0.0],
            [ s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
    

    def reset(self, eta0, nu0=None, b0=None):
        self.eta_hat = np.array(eta0, dtype=float).reshape(3)
        if nu0 is None:
            self.nu_hat  = np.zeros(3) 
        else: 
           self.nu_hat = np.array(nu0, dtype=float).reshape(3)

        if b0 is None:
            self.b_hat = np.zeros(3)
        else:
            self.b_hat = np.array(b0, dtype=float).reshape(3)

        self.eta_hat[2] = self.wrap_angle(self.eta_hat[2])

    def step(self, dt, eta_meas, tau):
        eta_meas = np.array(eta_meas, dtype=float).reshape(3)  # NED
        tau = np.array(tau, dtype=float).reshape(3)            # BODY
    
        eta_tilde = eta_meas - self.eta_hat
        eta_tilde[2] = self.wrap_angle(eta_tilde[2])

        y_tilde = eta_tilde

        psi = float(eta_meas[2])
        Rpsi = self.R(psi)

        eta_hat_dot = Rpsi @ self.nu_hat + self.L1 @ y_tilde

        nu_hat_dot = self.M_inv @ (-self.D @ self.nu_hat + self.b_hat + tau + self.L2 @ (Rpsi.T @ y_tilde))

        b_hat_dot = -self.Tb_inv @ self.b_hat + self.L3 @ (Rpsi.T @ y_tilde)

        self.eta_hat += dt * eta_hat_dot
        self.nu_hat  += dt * nu_hat_dot
        self.b_hat   += dt * b_hat_dot
        self.eta_hat[2] = self.wrap_angle(self.eta_hat[2])

        return self.eta_hat, self.nu_hat, self.b_hat


dt = 0.01
T  = 200.0
N  = int(T/dt) + 1
t  = np.linspace(0.0, T, N)
w = 1e-5 * np.ones(3)

obs = NPOObserver()

eta_tilde0 = np.array([1.0, 1.0, np.pi/4])
nu_tilde0  = np.array([0.1, 0.1, 0.0])
b_tilde0   = np.array([0.1, -0.1, 0.01])

eta_true = np.zeros(3)
nu_true  = np.zeros(3)
b_true   = np.zeros(3)

obs.reset(eta_true - eta_tilde0,
          nu_true  - nu_tilde0,
          b_true   - b_tilde0)

tau = np.zeros(3)

eta_tilde_log = np.zeros((N,3))
nu_tilde_log  = np.zeros((N,3))
b_tilde_log   = np.zeros((N,3))

for k, tk in enumerate(t):

    # psi(t) = 0.1 t (eksakt)
    eta_true[:] = 0.0
    eta_true[2] = obs.wrap_angle(0.1 * tk)

    eta_meas = eta_true + w
    eta_meas[2] = obs.wrap_angle(eta_meas[2])

    eta_hat, nu_hat, b_hat = obs.step(dt, eta_meas, tau)

    eta_tilde = eta_true - eta_hat
    eta_tilde[2] = obs.wrap_angle(eta_tilde[2])

    nu_true[:] = 0.0
    nu_true[2] = 0.1

    nu_tilde = nu_true - nu_hat
    b_tilde  = b_true - b_hat

    eta_tilde_log[k,:] = eta_tilde
    nu_tilde_log[k,:]  = nu_tilde
    b_tilde_log[k,:]   = b_tilde

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# --- eta tilde ---
axs[0].plot(t, eta_tilde_log[:,0], label=r'$\tilde{x}$')
axs[0].plot(t, eta_tilde_log[:,1], label=r'$\tilde{y}$')
axs[0].plot(t, eta_tilde_log[:,2], label=r'$\tilde{\psi}$')
axs[0].set_ylabel(r'$\tilde{\eta}$')
axs[0].grid(True)
axs[0].legend()

# --- nu tilde ---
axs[1].plot(t, nu_tilde_log[:,0], label=r'$\tilde{u}$')
axs[1].plot(t, nu_tilde_log[:,1], label=r'$\tilde{v}$')
axs[1].plot(t, nu_tilde_log[:,2], label=r'$\tilde{r}$')
axs[1].set_ylabel(r'$\tilde{\nu}$')
axs[1].grid(True)
axs[1].legend()

# --- b tilde ---
axs[2].plot(t, b_tilde_log[:,0], label=r'$\tilde{b}_x$')
axs[2].plot(t, b_tilde_log[:,1], label=r'$\tilde{b}_y$')
axs[2].plot(t, b_tilde_log[:,2], label=r'$\tilde{b}_\psi$')
axs[2].set_ylabel(r'$\tilde{b}$')
axs[2].set_xlabel('Time [s]')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()