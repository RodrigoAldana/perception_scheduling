import numpy as np
from target_model import Target


class Kalman:
    def __init__(self, target: Target):
        self.P = np.eye(target.ss_dim)
        self.x = np.zeros((target.ss_dim,1))
        self.t = 0
        self.target = target
        self.xp = self.x
        self.Pp = self.P

    def set_prior(self, x0, P0):
        self.P = P0
        self.x = x0

    def update(self, t, z, mode):
        self.t = t
        Ad = self.target.Ad[mode]
        C = self.target.C
        self.P, L = self.update_cov(self.P, mode)
        self.x = Ad @ self.x + L @ (z - C @ self.x)
        self.xp = self.x
        self.Pp = self.P

        return self.x, self.P

    def update_cov(self, P, mode):
        Ad = self.target.Ad[mode]
        Wd = self.target.Wd[mode]
        C = self.target.C
        R = self.target.mcovs[mode]
        L = Ad @ P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
        return (Ad - L @ C) @ P @ (Ad - L @ C).T + L @ R @ L.T + Wd, L



    def predict(self, t, override=False):
        Ad, Wd = self.target.transition_matrices(t-self.t)
        P = self.P
        x = self.x

        x = Ad @ x
        P = Ad @ P @ Ad + Wd

        if override:
            self.x = x
            self.P = P
        self.xp = x
        self.Pp = P
        return x, P






