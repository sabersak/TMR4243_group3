import numpy as np
import matplotlib.pyplot as plt
from luenberger_obs import LuenbergerObserver, Rz, wrap_pi

def band_limited_noise_step(x, sigma, alpha):
    w = np.random.randn(3) * sigma
    x = x + alpha * (w - x)
    return x

def simulate_errors(dt, T, L1, L2, L3, Tb, k, dr_window=None):
    M = LuenbergerObserver().M
    D = LuenbergerObserver().D
    M_inv = np.linalg.inv(M)
    Tb_inv = np.linalg.inv(Tb)

    N = int(T/dt) + 1
    t = np.linspace(0, T, N)

    # initial conditions (22)-(24)
    eta_t = np.array([1.0, 1.0, np.pi/4])
    nu_t  = np.array([0.1, 0.1, 0.0])
    b_t   = np.array([0.1, -0.1, 0.01])

    ETA = np.zeros((N,3))
    NU  = np.zeros((N,3))
    B   = np.zeros((N,3))
    YT  = np.zeros((N,3))
    DR  = np.zeros(N)

    # noise settings
    noise_power = 1e-5
    sigma = np.sqrt(noise_power)
    tau_lp = 0.2
    alpha = dt/(tau_lp + dt)
    nstate = np.zeros(3)

    for i in range(N):
        ti = t[i]
        psi = 0.1 * ti  # psi(t)=0.1t

        # band-limited noise
        nstate = band_limited_noise_step(nstate, sigma, alpha)
        w1, w2, w3 = nstate

        Rm = Rz(psi + w3)

        # innovation y_tilde = eta_tilde + w
        y_t = np.array([eta_t[0] + w1, eta_t[1] + w2, eta_t[2] + w3])

        # wrapped yaw innovation (Eq. 21)
        psi_meas = wrap_pi(psi + w3)
        psi_hat  = wrap_pi(psi - eta_t[2])    # eta_t[2] = psi - psi_hat
        y_t[2]   = wrap_pi(psi_meas - psi_hat)

        # dead reckoning window => disable injection
        if dr_window is not None:
            if dr_window[0] <= ti <= dr_window[1]:
                y_t[:] = 0.0
                DR[i] = 1.0

        # log
        ETA[i] = eta_t
        NU[i]  = nu_t
        B[i]   = b_t
        YT[i]  = y_t

        # Eq. (17)-(19) 
        eta_dot = Rm @ nu_t - (L1 @ y_t)
        nu_dot  = M_inv @ (-D @ nu_t + b_t - (k*L2) @ (Rm.T @ y_t))
        b_dot   = -Tb_inv @ b_t - (k*L3) @ (Rm.T @ y_t)

        eta_t = eta_t + dt*eta_dot
        nu_t  = nu_t  + dt*nu_dot
        b_t   = b_t   + dt*b_dot

        eta_t[2] = wrap_pi(eta_t[2])

    return t, ETA, NU, B, YT, DR

def plot3(t, X, titlestr, labels):
    plt.figure()
    plt.plot(t, X[:,0], label=labels[0])
    plt.plot(t, X[:,1], label=labels[1])
    plt.plot(t, X[:,2], label=labels[2])
    plt.grid(True)
    plt.title(titlestr)
    plt.xlabel("t [s]")
    plt.legend()

def main():
    dt = 0.01
    T  = 200.0

    Tb = 142.0*np.eye(3)

    # -------- Gains --------
    omega_c = 0.2
    L1 = omega_c*np.eye(3)
    L2 = (omega_c**2)*np.eye(3)
    L3 = 0.1*(omega_c**2)*np.eye(3)   
    k  = 1.0

    #dr_window = (60.0, 90.0)  # set None to disable
    dr_window = None

    t, ETA, NU, B, YT, DR = simulate_errors(dt, T, L1, L2, L3, Tb, k, dr_window)

    # main plots
    plot3(t, ETA, "eta_tilde", ["~x", "~y", "~psi"])
    plot3(t, NU,  "nu_tilde",  ["~u", "~v", "~r"])
    plot3(t, B,   "b_tilde",   ["~b1", "~b2", "~b3"])
    plot3(t, YT,  "innovation used (tilde y)", ["~y1", "~y2", "~y3"])

    if dr_window is not None:
        # show dead reckoning on/off
        plt.figure()
        plt.plot(t, DR)
        plt.ylim([-0.1, 1.1])
        plt.grid(True)
        plt.title("Dead reckoning flag (ON)")
        plt.xlabel("t [s]")

    plt.show()

if __name__ == "__main__":
    main()
