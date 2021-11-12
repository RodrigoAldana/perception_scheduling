import numpy as np
from scipy.linalg import block_diag, expm


class MultiTarget:
    def __init__(self):
        self.targets = []
        self.deltas = []

    # Adds integrator target.
    def add_target(self, number, order, ws_dim, pcov):
        for i in range(0, number):
            self.targets.append(Target(order, ws_dim, pcov))

    def set_random_x0(self):
        for target in self.targets:
            target.set_random_x0()

    def set_latency(self, deltas, mcovs):
        self.deltas = deltas
        for target in self.targets:
            target.set_latency(deltas, mcovs)


class Target:
    def __init__(self, order, ws_dim, pcov):

        self.integration_points = 100
        self.n_deltas = 0
        self.deltas = []
        self.Ad = []
        self.Wd = []
        self.mcovs = []
        self.mcovs_chol = []

        # Individual coordinate dynamical order
        self.order = order

        # Workspace dimension, usually 2 or 3
        self.ws_dim = ws_dim

        # Complete state space dimension
        self.ss_dim = order*ws_dim

        # Form of A,B,x0 for order-integrator of ws_dim dimensions
        self.x0 = np.zeros((self.ss_dim, 1))
        A = np.zeros((order, order))
        A[0:-1, 1:] = np.eye(order-1)
        B = np.zeros((order, 1))
        B[-1] = 1
        C = np.zeros((1, order))
        C[0] = 1
        self.A = A
        self.B = B
        self.C = C
        for d in range(0, self.ws_dim-1):
            self.A = block_diag(self.A, A)
            self.B = block_diag(self.B, B)
            self.C = block_diag(self.C, C)
        self.x = self.x0

        # covariance matrices input as diagonal matrices
        # OR as vector (converted to diagonal matrix)
        # OR as scalar, repeated as vector and converted to diagonal matrix
        if len(pcov.shape) == 1:
            if pcov.shape[0] == 1:
                self.pcov = np.diag(np.tile(pcov, ws_dim))
            else:
                self.pcov = np.diag(pcov)
        else:
            self.pcov = pcov

        # Obtain the equivalent to standard deviation as Cholesky factorization
        self.pcov_chol = np.linalg.cholesky(self.pcov)

    def set_random_x0(self):
        for i in range(0, self.ss_dim, self.order):
            self.x0[i] = np.random.randn(1)
        self.x = self.x0

    def step(self, h):
        # Implementation of Euler-Maruyama step:

        # Deterministic increment
        self.x = self.x + h*self.A @ self.x

        # Brownian increment
        dw = (self.B @ self.pcov_chol) @ np.random.randn(self.pcov.shape[0], 1)

        # Contribution of both increments
        self.x = self.x + np.sqrt(h) * dw
        return self.x

    def set_latency(self, deltas, mcovs):
        self.n_deltas = len(deltas)
        self.deltas = deltas
        self.Ad = []
        self.mcovs = []
        self.mcovs_chol = []
        for i, latency in enumerate(deltas):
            Ad, Wd = self.transition_matrices(latency)
            self.Ad.append(Ad)
            self.Wd.append(Wd)
            mcov = mcovs[i]
            if len(mcov.shape) == 1:
                if mcov.shape[0] == 1:
                    mcov = np.diag(np.tile(mcov, self.ws_dim))
                else:
                    mcov = np.diag(mcov)
            self.mcovs.append(mcov)
            self.mcovs_chol.append(np.linalg.cholesky(mcov))

    def transition_matrices(self, latency):
        Ad = expm(self.A*latency)
        n = self.integration_points
        t = np.linspace(0, latency, n)
        dt = latency/float(n)
        Wd = np.zeros(Ad.shape)
        for i, ti in enumerate(t):
            Adt = expm(self.A*ti)
            Wd = Wd + Adt @ self.B @ self.pcov @ self.B.T @ Adt.T
        Wd = Wd*dt
        return Ad, Wd

    def sim_measure(self, mode):
        z = self.C @ self.x
        z = z + self.mcovs_chol[mode] @ np.random.randn(z.shape[0], 1)
        return z


