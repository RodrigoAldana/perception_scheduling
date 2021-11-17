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
                    Ad, Wd = self.target.Ad[mode], self.target.Wd[mode]
                    J = np.trace(Ad @ Pq @ Ad.T + Wd)*self.target.deltas[mode]
                    self.G[q].costs.append(J)



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

    def save_graph(self, name='graph', folder=''):
        save_object(self, folder+'/'+name)

    def dp_search(self, q0, Tf, r_pen, lambda_a, heuristic=False, silent=True):
        q0 = self.G[q0].index
        min_delta = min(self.target.deltas)
        steps = np.floor(np.array(self.target.deltas)/min_delta)
        deltas = min_delta*steps
        amax = int(Tf / min_delta)
        Qd = len(self.G)
        MQ = np.zeros((Qd, amax + 1))
        MP = np.zeros((Qd, amax + 1))
        MJ = np.inf*np.ones((Qd, amax+1))
        MK = np.zeros((Qd, amax + 1))

        MJ[q0, 0] = 0

        progress = ProgressBar('Computing DP matrices', amax, interval=5, silent=silent)
        for k in range(0, amax+1):
            progress.print_progress(k)
            for q in range(0, Qd):
                qreal = list(self.G.keys())[q]
                Pq = self.G[qreal].Pq
                Nq = self.G[qreal].options
                if heuristic and self.G[qreal].preferred_scheduling != -1:
                    p = self.G[qreal].preferred_scheduling
                    p_range = range(p, p+1)
                else:
                    p_range = range(0, len(Nq))
                for p in p_range:
                    qp = Nq[p]
                    qp = self.G[qp].index
                    tau = k*min_delta
                    tau_plus = tau + deltas[p]
                    tau_plus = (k + steps[p])*min_delta
                    Tmax = np.min((tau_plus, Tf))
                    if(tau_plus < Tf):
                        J = self.G[qreal].costs[p]
                    else:
                        Ad, Wd = self.target.transition_matrices(Tmax - tau)
                        J = np.trace(Ad @ Pq @ Ad.T + Wd)*(Tmax - tau)
                    J = (J + lambda_a*r_pen[p])/Tf
                    k_next = int(min(amax, k + steps[p]))
                    if MJ[q, k] + J < MJ[qp, k_next]:
                        MJ[qp, k_next] = MJ[q, k] + J
                        MQ[qp, k_next] = q
                        MP[qp, k_next] = p
                        MK[qp, k_next] = k
        J_opt = np.Inf
        q = 0
        for qp in range(0, Qd):
            if MJ[qp, amax] < J_opt:
                q = qp
                J_opt = MJ[qp, amax]

        q_opt = [q]
        p_opt = []
        k = int(amax)
        while k != 0:
            q_opt.append(int(MQ[q, k]))
            p_opt.append(int(MP[q, k]))
            k = int(MK[q, k])
            q = q_opt[-1]

        p_opt = p_opt[::-1]
        q_opt = q_opt[::-1]
        return p_opt, q_opt, J_opt

    def fill_preferred_schedules(self, Tf, r_pen, lambda_a, heuristic=False):
        progress = ProgressBar('Filling schedules', len(list(self.G.keys())), interval=0.1)
        for index, key in enumerate(list(self.G.keys())):
            progress.print_progress(current=index)
            p_opt, _, _ = self.dp_search(q0=key, Tf=Tf, r_pen=r_pen, lambda_a=lambda_a, heuristic=heuristic)
            self.G[key].preferred_scheduling = p_opt[0]
            # print(("Preferred schedule:", p_opt[0]))
            # print(self.G[key].Pq)

    def cost_of_schedule(self, schedule, q0, Tf, r_pen, lambda_a):
        q = q0
        J = 0
        deltas = self.target.deltas
        tau = 0
        finish = False
        for i, p in enumerate(schedule):
            # print((p, J, q))
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
        self.preferred_scheduling = -1
        self.costs = []



