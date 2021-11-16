import numpy as np
from target_kalman import Kalman
from utils import ProgressBar, save_object
from tictoc import tic, toc

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

        for index, key in enumerate(self.G.keys()):
            node = self.G[key]
            node.index = index




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

    def dp_search(self, q0, Tf, r_pen, lambda_a):
        q0 = self.G[q0].index
        deltas = self.target.deltas
        amax = int(Tf / np.min(deltas))
        Qd = len(self.G)
        MQ = np.zeros((Qd, amax + 1))
        MP = np.zeros((Qd, amax + 1))
        MJ = np.inf*np.ones((Qd, amax+1))
        MT = np.zeros((Qd, amax + 1))
        Ma = amax*np.ones(Qd)

        MJ[q0, 0] = 0

        progress = ProgressBar('Computing DP matrices', amax, interval=5)
        for k in range(1, amax+1):
            progress.print_progress(k)
            if k == 1:
                V = range(q0, q0+1)
            else:
                V = range(0, Qd)

            for q in V:
                qreal = list(self.G.keys())[q]
                Pq = self.G[qreal].Pq
                Nq = self.G[qreal].options
                for p in range(0, len(Nq)):
                    qp = Nq[p]
                    qp = self.G[qp].index
                    tau = MT[q, k-1]
                    tau_plus = tau + deltas[p]
                    Tmax = np.min((tau_plus, Tf))
                    if(tau_plus < Tf):
                        Ad, Wd = self.target.Ad[p], self.target.Wd[p]
                    else:
                        Ad, Wd = self.target.transition_matrices(Tmax - tau)
                    J = np.trace(Ad @ Pq @ Ad.T + Wd)*(Tmax - tau)
                    J = (J + lambda_a*r_pen[p])/Tf
                    if MJ[q, k-1] + J < MJ[qp, k]:
                        MJ[qp, k] = MJ[q, k-1] + J
                        MQ[qp, k] = q
                        MT[qp, k] = Tmax
                        MP[qp, k] = p
                        ap = int(Ma[qp])
                        if tau_plus >= Tf and MJ[qp, k] < MJ[qp, ap]:
                            Ma[qp] = k
        print(MJ)
        J_opt = np.inf
        for q in range(0, Qd):
            a = int(Ma[q])
            if MJ[q, a] < J_opt:
                J_opt = MJ[q, a]
                q_opt = q
                a_opt = a
        k = a_opt
        traj_opt = [q_opt]
        p_opt = []
        while k > 0:
            p_opt.append(int(MP[q_opt, k]))
            q_opt = int(MQ[q_opt, k])
            traj_opt.append(q_opt)
            k = k - 1
        traj_opt = traj_opt[::-1]  # Flip q_opt
        p_opt = p_opt[::-1]  # Flip p_opt

        keys = list(self.G.keys())
        print(traj_opt)
        print(keys)

        return p_opt, J_opt

    def cost_of_schedule(self, schedule, q0, Tf, r_pen, lambda_a):
        q = q0
        J = 0
        deltas = self.target.deltas
        tau = 0
        finish = False
        for i, p in enumerate(schedule):
            print((p, J, q))
            Pq = self.G[q].Pq
            tau_plus = tau + deltas[p]
            Tmax = np.min((tau_plus, Tf))
            if tau_plus < Tf:
                Ad, Wd = self.target.Ad[p], self.target.Wd[p]
            else:
                Ad, Wd = self.target.transition_matrices(Tmax - tau)
                finish = True
            Jp = np.trace(Ad @ Pq @ Ad.T + Wd) * (Tmax - tau)
            Jp = (Jp + lambda_a * r_pen[p]) / Tf
            J = Jp + J
            tau = tau_plus
            q = self.G[q].options[p]

            if finish:
                break
            if i == len(schedule)-1:
                schedule.append(schedule[-1])
        return J



class CovarianceNode:
    def __init__(self, Pq):
        self.Pq = Pq
        self.options = []
        self.index = 0



