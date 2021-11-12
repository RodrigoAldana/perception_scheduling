from covariance_graph import CovarianceGraph
import numpy as np
from target_model import MultiTarget


pcov = np.array([0.01])
deltas = [0.1, 0.5]
mcovs = [np.array([0.01]),np.array([0.001])]

mt = MultiTarget()
mt.add_target(number=1, order=2, ws_dim=1, pcov=pcov)
mt.set_random_x0()
mt.set_latency(deltas, mcovs)

cg = CovarianceGraph(mt.targets[0], step=0.5, bound=1)
cg.save_graph()

print('END')


