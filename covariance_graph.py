import numpy as np
from target_kalman import Kalman
from utils import ProgressBar, save_object
from tictoc import tic, toc


class CovarianceGraph:
    def __init__(self, target, step, bound, exhaustive=True, n_samples=1, max_trials=100):
        self.exhaustive = exhaustive
        self.target = target
        self.step = step
        self.bound = np.round(bound / step) * step
        self.n_steps = np.round(bound / step)
        self.ss_dim = self.target.ss_dim
        self.kalman = Kalman(target)
        self.n_deltas = self.target.n_deltas
        self.max_patch_distance = -np.Inf
        self.G = dict()
        self.extra_states = []
        self.max_trials = max_trials
        self.preferred_schedules = {-1: {-1: 0}}
        self.min_delta = min(self.target.deltas)
        self.delta_steps = np.floor(np.array(self.target.deltas)/self.min_delta).astype(int)-1
        self.deltas = self.min_delta*self.delta_steps
        if exhaustive:
            self.create_exhaustive_graph()
        else:
            self.create_sampled_graph(n_samples)
        self.correct_spurious()

        for index, key in enumerate(self.G.keys()):
            node = self.G[key]
            node.index = index

    def create_node(self, q, Pq, force_spurious=False):
        self.G[q] = CovarianceNode(Pq)
        for mode in range(0, self.n_deltas):
            P_mode, _ = self.kalman.update_cov(Pq, mode)
            P_mode_q, q_mode = self.quantize(P_mode)
            self.G[q].options.append(q_mode)
            if ~self.is_pos_semi_def(P_mode_q) or force_spurious:
                self.extra_states.append({'q': q, 'mode': mode, 'Pq': P_mode})
            Ad, Wd = self.target.Ad[mode], self.target.Wd[mode]
            J = np.trace(Ad @ Pq @ Ad.T + Wd) * self.target.deltas[mode]
            self.G[q].costs.append(J)
        for i in range(0, np.max(self.delta_steps)+1):
            Ad, Wd = self.target.Ad_steps[i], self.target.Wd_steps[i]
            J = np.trace(Ad @ Pq @ Ad.T + Wd) * (i+1)*self.min_delta
            self.G[q].partial_costs.append(J)


    def create_exhaustive_graph(self):
        maximal_combs = int((2 * self.n_steps) ** (self.ss_dim ** 2))
        progress = ProgressBar('Building graph', maximal_combs)
        for q in range(0, maximal_combs):
            progress.print_progress(q)
            Pq = self.get_Pq(q)
            if self.is_pos_semi_def(Pq):
                self.create_node(q, Pq)

    def create_sampled_graph(self, n_samples):
        maximal_combs = int((2 * self.n_steps) ** (self.ss_dim ** 2))
        n_samples = min(n_samples, maximal_combs)
        progress = ProgressBar('Building graph', n_samples)
        current_elements = 0
        trials = 0
        while current_elements < n_samples:
            progress.print_progress(current_elements)
            P = 2*(np.random.rand(self.ss_dim, self.ss_dim)-0.5)
            P = self.bound*(P.T @ P)
            Pq, q = self.quantize(P)
            if self.is_pos_semi_def(Pq) and q not in self.G:
                current_elements = current_elements + 1
                trials = 0
                self.create_node(q, Pq, force_spurious=True)
            else:
                trials = trials + 1
            if trials > self.max_trials:
                break

    def correct_spurious(self):
        progress = ProgressBar('Correcting spurious', len(self.extra_states))
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

    def closest(self, P):
        if self.exhaustive:
            Pq, q = self.quantize(P)
        else:
            min_distance = np.Inf
            best_q = 0
            for q, node in self.G.items():
                distance = np.sqrt(np.sum(np.abs(P - node.Pq)))
                if distance < min_distance:
                    best_q = q
                    min_distance = distance
            q = best_q
            Pq = self.G[q].Pq
        return Pq, q

    def quantize(self, P):
        Pq = np.clip(P, -self.bound, self.bound - self.step/2)
        q = np.floor(Pq / self.step)
        Pq = q * self.step + self.step/2
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
        Pq = Pq*self.step + self.step/2
        Pq.resize((self.ss_dim,self.ss_dim))
        return Pq

    def save_graph(self, name='graph', folder='graphs'):
        save_object(self, name, folder)

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
                    if(tau_plus <= Tf):
                        J = self.G[qreal].costs[p]
                    else:
                        step = int(amax-k)
                        if tau < Tf:
                            J = self.G[qreal].partial_costs[step]
                        else:
                            J = 0
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

    def dp_search_2d(self, q0, Tf, r_pen, lambda_a, heuristic=False, silent=True):
        q0x = self.G[q0[0]].index
        q0y = self.G[q0[1]].index
        min_delta = min(self.target.deltas)
        steps = np.floor(np.array(self.target.deltas) / min_delta)
        deltas = min_delta * steps
        amax = int(Tf / min_delta)
        Qd = len(self.G)
        MQx = np.zeros((Qd, Qd, amax + 1))
        MQy = np.zeros((Qd, Qd, amax + 1))
        MP = np.zeros((Qd, Qd, amax + 1))
        MJ = np.inf * np.ones((Qd, Qd, amax + 1))
        MK = np.zeros((Qd, Qd, amax + 1))

        MJ[q0x, q0y, 0] = 0

        progress = ProgressBar('Computing DP matrices', amax, interval=5, silent=silent)
        for k in range(0, amax + 1):
            progress.print_progress(k)
            for qx in range(0, Qd):
                for qy in range(0, Qd):
                    qrealx = list(self.G.keys())[qx]
                    qrealy = list(self.G.keys())[qy]
                    Pqx = self.G[qrealx].Pq
                    Pqy = self.G[qrealy].Pq
                    Nqx = self.G[qrealx].options
                    Nqy = self.G[qrealy].options
                    if heuristic and self.preferred_schedules[qx][qy] != -1:
                        p = self.preferred_schedules[qx][qy]
                        p_range = range(p, p + 1)
                    else:
                        p_range = range(0, self.n_deltas)

                    for p in p_range:
                        qpx = Nqx[p]
                        qpy = Nqy[p]
                        qpx = self.G[qpx].index
                        qpy = self.G[qpy].index
                        tau = k * min_delta
                        tau_plus = tau + deltas[p]
                        tau_plus = (k + steps[p]) * min_delta
                        Tmax = np.min((tau_plus, Tf))
                        if (tau_plus <= Tf):
                            J = self.G[qrealx].costs[p]
                            J = J + self.G[qrealy].costs[p]
                        else:
                            step = int(amax - k)
                            if tau < Tf:
                                J = self.G[qrealx].partial_costs[step]
                                J = J + self.G[qrealy].partial_costs[step]
                            else:
                                J = 0
                        J = (J + lambda_a * r_pen[p]) / Tf
                        k_next = int(min(amax, k + steps[p]))
                        if MJ[qx, qy, k] + J < MJ[qpx, qpy, k_next]:
                            MJ[qpx, qpy, k_next] = MJ[qx, qy, k] + J
                            MQx[qpx, qpy, k_next] = qx
                            MQy[qpx, qpy, k_next] = qy
                            MP[qpx, qpy, k_next] = p
                            MK[qpx, qpy, k_next] = k
        J_opt = np.Inf
        qx = 0
        qy = 0
        for qpx in range(0, Qd):
            for qpy in range(0, Qd):
                if MJ[qpx, qpy, amax] < J_opt:
                    qx = qpx
                    qy = qpy
                    J_opt = MJ[qpx, qpy, amax]

        q_opt = [(qx, qy)]
        p_opt = []
        k = int(amax)
        while k != 0:
            q_opt.append((int(MQx[qx, qy, k]), int(MQy[qx, qy, k])))
            p_opt.append(int(MP[qx, qy, k]))
            k = int(MK[qx, qy, k])
            qx = q_opt[-1][0]
            qy = q_opt[-1][1]

        p_opt = p_opt[::-1]
        q_opt = q_opt[::-1]
        self.preferred_schedules[q0[0]] = {q0[1]: p_opt[0]}
        return p_opt, q_opt, J_opt

    def fill_preferred_schedules(self, Tf, r_pen, lambda_a, heuristic=False):
        progress = ProgressBar('Filling schedules', len(list(self.G.keys())), interval=0.1)
        for index, key in enumerate(list(self.G.keys())):
            progress.print_progress(current=index)
            p_opt, _, _ = self.dp_search(q0=key, Tf=Tf, r_pen=r_pen, lambda_a=lambda_a, heuristic=heuristic)
            self.G[key].preferred_scheduling = p_opt[0]
            # print(("Preferred schedule:", p_opt[0]))
            # print(self.G[key].Pq)

    def fill_preferred_schedules_2d(self, Tf, r_pen, lambda_a, heuristic=False):
        progress = ProgressBar('Filling schedules', len(list(self.G.keys()))*len(list(self.G.keys())), interval=0.1)
        c = 0
        for keyx in (list(self.G.keys())):
            for keyy in (list(self.G.keys())):
                c = c + 1
                progress.print_progress(current=c)
                self.dp_search_2d(q0=(keyx, keyy), Tf=1, r_pen=r_pen, lambda_a=lambda_a, silent=False)

    def cost_of_schedule(self, schedule, q0, Tf, r_pen, lambda_a):
        q = q0
        J = 0
        deltas = self.target.deltas
        tau = 0
        finish = False
        for i, p in enumerate(schedule):
            # prints((p, J, q))
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

    def real_cost_of_schedule(self, schedule, q0, Tf, r_pen, lambda_a):
        q = q0
        P = self.G[q].Pq
        J = 0
        deltas = self.target.deltas
        tau = 0
        finish = False
        for i, p in enumerate(schedule):
            tau_plus = tau + deltas[p]
            Tmax = np.min((tau_plus, Tf))
            if tau_plus < Tf:
                Ad, Wd = self.target.Ad[p], self.target.Wd[p]
            else:
                Ad, Wd = self.target.transition_matrices(Tmax - tau)
                finish = True
            Jp = np.trace(Ad @ P @ Ad.T + Wd) * (Tmax - tau)
            Jp = (Jp + lambda_a * r_pen[p]) / Tf
            J = Jp + J
            tau = tau_plus
            P, _ = self.kalman.update_cov(P, p)
            if finish:
                break
            if i == len(schedule)-1:
                schedule.append(schedule[-1])
        return J

    def cost_of_schedule_2d(self, schedule, q0, Tf, r_pen, lambda_a):
        qx = q0[0]
        qy = q0[1]
        J = 0
        deltas = self.target.deltas
        tau = 0
        finish = False
        for i, p in enumerate(schedule):
            # prints((p, J, q))
            Pqx = self.G[qx].Pq
            Pqy = self.G[qy].Pq
            tau_plus = tau + deltas[p]
            Tmax = np.min((tau_plus, Tf))
            if tau_plus < Tf:
                Ad, Wd = self.target.Ad[p], self.target.Wd[p]
            else:
                #Ad, Wd = self.target.Ad[p], self.target.Wd[p]
                Ad, Wd = self.target.transition_matrices(Tmax - tau)
                finish = True
            Jp = np.trace(Ad @ Pqx @ Ad.T + Wd) * (Tmax - tau)
            Jp = Jp + np.trace(Ad @ Pqy @ Ad.T + Wd) * (Tmax - tau)
            Jp = (Jp + lambda_a * r_pen[p]) / Tf
            J = Jp + J
            tau = tau_plus
            qx = self.G[qx].options[p]
            qy = self.G[qy].options[p]

            if finish:
                break
            if i == len(schedule)-1:
                schedule.append(schedule[-1])
        return J

    def real_cost_of_schedule_2d(self, schedule, q0, Tf, r_pen, lambda_a):
        qx = q0[0]
        qy = q0[1]
        Px = self.G[qx].Pq
        Py = self.G[qy].Pq
        J = 0
        deltas = self.target.deltas
        tau = 0
        finish = False
        for i, p in enumerate(schedule):
            tau_plus = tau + deltas[p]
            Tmax = np.min((tau_plus, Tf))
            if tau_plus < Tf:
                Ad, Wd = self.target.Ad[p], self.target.Wd[p]
            else:
                #Ad, Wd = self.target.Ad[p], self.target.Wd[p]
                Ad, Wd = self.target.transition_matrices(Tmax - tau)
                finish = True
            Jp = np.trace(Ad @ Px @ Ad.T + Wd) * (Tmax - tau)
            Jp = Jp + np.trace(Ad @ Py @ Ad.T + Wd) * (Tmax - tau)
            Jp = (Jp + lambda_a * r_pen[p]) / Tf
            J = Jp + J
            tau = tau_plus
            Px, _ = self.kalman.update_cov(Px, p)
            Py, _ = self.kalman.update_cov(Py, p)

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
        self.partial_costs = []



