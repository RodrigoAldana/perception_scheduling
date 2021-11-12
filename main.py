import numpy as np
from target_sim import TargetSim
from viz_utils import plot2d_order2


sim = TargetSim(skip_sim=50)
pcov = np.array([0.01])
sim.add_target(number=1, order=2, ws_dim=2, pcov=pcov)
sim.set_random_x0()

h = 0.001
deltas = [0.1]
mcovs = [np.array([0.001])]


t_window = 10

t, x, xp, P = sim.q_sampling_sim(h, t_window, deltas, mcovs)

plot2d_order2(t, x, xp, P)



