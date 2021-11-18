from source.covariance_graph import CovarianceGraph
import numpy as np
from source.target_model import MultiTarget

pcov = np.array([1])
deltas = [0.2, 0.1]
mcovs = [np.array([0.5]), np.array([0.005])]

mt = MultiTarget()
mt.add_target(number=1, order=2, ws_dim=1, pcov=pcov)
mt.set_random_x0()
mt.set_latency(deltas, mcovs)

cg = CovarianceGraph(mt.targets[0], step=0.2, bound=2)

cg2 = CovarianceGraph(mt.targets[0], step=0.2, bound=2, exhaustive=False, n_samples=100)

print('END')

