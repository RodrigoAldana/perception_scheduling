import numpy as np
from target_model import MultiTarget
from target_kalman import Kalman
from copy import deepcopy
from tictoc import tic, toc


class DefaultPerception:
    def __init__(self, targets: MultiTarget, deltas=None, mcovs=None):
        self.system = targets
        if deltas is not None:
            self.set_measure_params(deltas, mcovs)

    def set_measure_params(self, deltas, mcovs):
        self.system.set_latency(deltas, mcovs)

    def timer_handle(self, timer):
        return timer

    def inter_event_prediction(self, target_id, timer):
        x = self.system.targets[target_id].x
        P = np.zeros((len(x), len(x)))
        return x, P


class StaticPerception(DefaultPerception):
    def __init__(self, targets, deltas, mcovs):
        super().__init__(targets, deltas, mcovs)
        self.filters = []
        self.past_system = deepcopy(self.system)
        self.skip_counter = 0
        for target in targets.targets:
            kalman_filter = Kalman(target)
            self.filters.append(kalman_filter)
        self.perception_mode = 0

    def schedule(self, timer, measurements):
        pass

    def timer_handle(self, timer, real_measurements=None):
        if timer > self.system.deltas[self.perception_mode]:
            timer = 0
            if real_measurements is None:
                targets = self.past_system.targets
                for i, target in enumerate(targets):
                    z = target.sim_measure(self.perception_mode)
                    self.filters[i].update(timer, z, self.perception_mode)
                self.past_system = deepcopy(self.system)
                self.schedule(timer, real_measurements)
            else:
                for i, z in enumerate(real_measurements):
                    self.filters[i].update(timer, z, self.perception_mode)

        return timer

    def inter_event_prediction(self, target_id, timer, skip=-1):
        if self.skip_counter > skip != -1:
            x, P = self.filters[target_id].predict(timer, override=False)
            self.skip_counter = 0
        else:
            x, P = self.filters[target_id].xp, self.filters[target_id].Pp
            self.skip_counter = self.skip_counter + 1
        return x, P


class QPerception(StaticPerception):
    def __init__(self, targets, deltas, mcovs):
        super().__init__(targets, deltas, mcovs)

    def schedule(self, timer, measurements):
        pass

