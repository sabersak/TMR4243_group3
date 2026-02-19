import numpy as np
import matplotlib.pyplot as plt

# Viktig: importer klassen (ikke bare modulen)
from NPO import NPOObserver

dt = 0.01
T  = 200.0
N  = int(T/dt) + 1
t  = np.linspace(0.0, T, N)
w = 0 * 1e-5 * np.ones(3)

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