import numpy as np
from target_kalman import Kalman
from utils import ProgressBar, load_object, save_object


class CovarianceGraph:
    def __init__(self, target, step, bound):
        self.target = target
        self.step = step
        self.bound = np.round(bound / step) * step
        self.n_steps = np.round(bound / step)
        self.ss_dim = self.target.ss_dim
        self.kalman = Kalman(target)
        self.n_deltas = self.target.n_deltas

        maximal_combs = int((2*self.n_steps)**(self.ss_dim**2))

        self.G = dict()
        self.extra_states = []
        progress = ProgressBar('Building graph', maximal_combs)
        for q in range(0, maximal_combs):
            progress.print_progress(q)
            Pq = self.get_Pq(q)
            if self.is_pos_semi_def(Pq):
                self.G[q] = CovarianceNode(Pq)
                for mode in range(0, self.n_deltas):
                    P_mode, _  = self.kalman.update_cov(Pq, mode)
                    P_mode_q, q_mode = self.quantize(P_mode)
                    self.G[q].options.append(q_mode)
                    if ~self.is_pos_semi_def(P_mode_q):
                        self.extra_states.append({'q': q, 'mode': mode, 'Pq': P_mode})

        progress = ProgressBar('Correcting spurious', len(self.extra_states))
        self.max_patch_distance = -np.Inf
        for i, state in enumerate(self.extra_states):
            progress.print_progress(i)
            Pq = state['Pq']
            min_distance = np.Inf
            best_q = 0
            for q, node in self.G.items():
                distance = np.sqrt(np.sum(np.abs(Pq - node.Pq)))
                if distance < min_distance:
                    best_q = q
                    min_distance = distance
            self.G[state['q']].options[state['mode']] = best_q
            if self.max_patch_distance < min_distance:
                self.max_patch_distance = min_distance




    def quantize(self, P):
        Pq = np.clip(P, -self.bound, self.bound)
        q = np.floor(P / self.step)
        Pq = q * self.step
        q = q.flatten()
        q = q + np.ones(len(q))*self.n_steps
        l = len(q)
        for i in range(0, l):
            q[i] = q[i]*(2*self.n_steps)**i
        q = int(np.sum(q))
        return Pq, q

    def get_id(self, Pq):
        Pq, q = self.quantize(Pq)
        return q

    def is_pos_semi_def(self, Pq):
        return np.all(np.linalg.eigvals(Pq) >= 0) and self.is_sym(Pq)

    def is_sym(self, Pq):
        r = np.sum(np.abs((Pq - Pq.T).flatten()))
        return r < self.step

    def get_Pq(self, q):
        digits = []
        for i in range(self.ss_dim*self.ss_dim-1,-1,-1):
            d = int(q / ((2*self.n_steps)**i))
            digits.append(d)
            q = q - d*((2*self.n_steps)**i)
        digits = np.flip(digits)
        Pq = np.array(digits) - np.ones(len(digits))*self.n_steps
        Pq = Pq*self.step
        Pq.resize((self.ss_dim,self.ss_dim))
        return Pq

    def save_graph(self, name='graph'):
        save_object(self, name)




class CovarianceNode:
    def __init__(self, Pq):
        self.Pq = Pq
        self.options = []



