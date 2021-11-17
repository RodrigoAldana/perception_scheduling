import numpy as np
from target_sim import TargetSim
from viz_utils import plot1d_order2
from utils import load_object

sim = TargetSim(skip_sim=50)
cg = load_object('graphs/graph11_17_2021-11_37_01')
sim.targets.targets.append(cg.target)
sim.set_random_x0()

h = 0.001
deltas = cg.target.deltas
mcovs = cg.target.mcovs

t_window = 10

t, x, xp, P = sim.q_sampling_sim(h, t_window, cg)

plot1d_order2(t, x, xp, P)




