from covariance_graph import CovarianceGraph
from utils import load_object
import numpy as np
from target_model import MultiTarget
from tictoc import tic, toc

np.set_printoptions(linewidth=200)
pcov = np.array([1])
deltas = [0.2, 0.1]
mcovs = [np.array([0.5]), np.array([0.005])]

mt = MultiTarget()
mt.add_target(number=1, order=2, ws_dim=1, pcov=pcov)
mt.set_random_x0()
mt.set_latency(deltas, mcovs)

cg = CovarianceGraph(mt.targets[0], step=0.2, bound=2, exhaustive=False, n_samples=100)

P0 = np.array([[0.9, 0.2], [0.2, 0.9]])
P0, q0 = cg.closest(P0)

pen = np.array([1, 1])
lambda_a = 0.1

cg.fill_preferred_schedules_2d(Tf=1, r_pen=pen, lambda_a =lambda_a)

cg.save_graph()