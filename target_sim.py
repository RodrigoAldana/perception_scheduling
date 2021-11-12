import numpy as np
from target_model import MultiTarget
from perception_scheduling import DefaultPerception, StaticPerception, QPerception
from copy import deepcopy


class TargetSim:
    def __init__(self, skip_sim):
        self.targets = MultiTarget()
        self.dummy_perception = DefaultPerception(self.targets, None, None)
        self.skip_sim = skip_sim

    def add_target(self, number, order, ws_dim, pcov):
        self.targets.add_target(number, order, ws_dim, pcov)

    def set_random_x0(self):
        self.targets.set_random_x0()

    # Raw simulation without any particular hybrid behaviour, just
    # Euler-Maruyama approximation of dynamics
    def raw_sim(self, h, t_window, perception=None):
        if perception is None:
            perception = self.dummy_perception

        t_window = h*np.floor(t_window/h)
        time = np.linspace(0, t_window, int(t_window/h))
        trajectories = []
        covariance_trajectories = []
        timer = 0

        # Initialize a state array per target
        for target in self.targets.targets:
            trajectories.append(np.zeros((target.ss_dim, len(time))))
            covariance_trajectories.append(np.zeros(( len(time), target.ss_dim, target.ss_dim)))
            trajectories[-1][:, 0] = target.x.reshape(target.ss_dim)

        estimated_trajectories = deepcopy(trajectories)

        # Update each target dynamics
        for t in range(1, len(time)):
            for i, target in enumerate(self.targets.targets):
                trajectories[i][:, t] = target.step(h).reshape(target.ss_dim)
                x_pred, P_pred = perception.inter_event_prediction(i, timer, skip=self.skip_sim)
                estimated_trajectories[i][:, t] = x_pred.reshape(target.ss_dim)
                covariance_trajectories[i][t, :, :] = P_pred
            timer = timer + h

            timer = perception.timer_handle(timer, real_measurements=None)
        return time, trajectories, estimated_trajectories, covariance_trajectories

    def static_sampling_sim(self, h, t_window, delta, mcov):
        perception = StaticPerception(self.targets, delta, mcov)
        return self.raw_sim(h, t_window, perception)

    def q_sampling_sim(self, h, t_window, delta, mcov):
        perception = QPerception(self.targets, delta, mcov)
        return self.raw_sim(h, t_window, perception)


