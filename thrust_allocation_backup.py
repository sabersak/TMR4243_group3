import numpy as np


class ThrustAllocator:

    def __init__(self):

        lx1 = -0.415
        ly1 = -0.07
        lx2 = -0.415
        ly2 = 0.07
        lx3 = 0.37

        self.B = np.array([      
        [1,       0,       1,       0,       0], 
        [0,       1,       0,       1,       1],
        [-ly1,    lx1,     -ly2,    lx2,     lx3]
        ], dtype='float')

        self.K = np.diag([1.0, 1.0, 1.0])

        self.W = np.diag([1.0, 1.0, 1.0, 1.0, 4.0])

        self.f_d = np.zeros(5)

    def weighted_pinv(self):
        #" B_W^† = W^{-1} B^T (B W^{-1} B^T)^-1 "
        W_inv = np.linalg.inv(self.W)
        return W_inv @ self.B.T @ np.linalg.pinv(self.B @ W_inv @ self.B.T)

    def clip_u(self, u):
        u[0] = np.clip(u[0], 0.0, 1.0)    # azimuth 1
        u[1] = np.clip(u[1], 0.0, 1.0)    # azimuth 2
        u[2] = np.clip(u[2], -1.0, 1.0)   # tunnel
        return u

    def rebuild_f(self, F, alpha):
        F1, F2, F3 = F
        alpha1, alpha2 = alpha

        X1 = F1 * np.cos(alpha1)
        Y1 = F1 * np.sin(alpha1)

        X2 = F2 * np.cos(alpha2)
        Y2 = F2 * np.sin(alpha2)

        Y3 = F3

        return np.array([X1, Y1, X2, Y2, Y3], dtype=float)

    def wrap_to_pi(self, angle):
        """Wrap angle(s) to [-pi, pi)"""
        return (angle + np.pi) % (2*np.pi) - np.pi
    
    def is_saturated(self, u, u_cmd, tol=1e-6):
        return np.any(np.abs(u - u_cmd) > tol)


    def allocate(self, tau_cmd):
        B_W_pinv = self.weighted_pinv()
        Q_W = np.eye(5) - B_W_pinv @ self.B

        self.f_star = B_W_pinv @ tau_cmd + Q_W @ self.f_d
        self.f_comp_cmd = np.array([self.f_star[0], self.f_star[1], self.f_star[2], self.f_star[3], self.f_star[4]])

        F_1 = np.sqrt(self.f_star[0]**2 + self.f_star[1]**2)
        F_2 = np.sqrt(self.f_star[2]**2 + self.f_star[3]**2)
        F_3 = self.f_star[4]

        alpha_1 = np.arctan2(self.f_star[1], self.f_star[0])
        alpha_2 = np.arctan2(self.f_star[3], self.f_star[2])
        alpha_1 = self.wrap_to_pi(alpha_1)
        alpha_2 = self.wrap_to_pi(alpha_2)

        F_cmd = np.array([F_1, F_2, F_3], dtype='float')
        alpha_cmd = np.array([alpha_1, alpha_2], dtype='float')
        u_cmd = np.linalg.inv(self.K) @ F_cmd

        self.f_d = self.f_comp_cmd
        u_cmd = self.clip_u(u_cmd)
    
        return F_cmd, alpha_cmd, u_cmd
    