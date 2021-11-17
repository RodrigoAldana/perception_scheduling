from source.covariance_graph import CovarianceGraph
from utils import load_object
from tictoc import tic, toc
import numpy as np
from source.target_model import MultiTarget

np.set_printoptions(linewidth=200)

pcov = np.array([0.1])
deltas = [0.2, 0.1]
mcovs = [np.array([0.1]), np.array([1])]

mt = MultiTarget()
mt.add_target(number=1, order=2, ws_dim=1, pcov=pcov)
mt.set_random_x0()
mt.set_latency(deltas, mcovs)

cg = CovarianceGraph(mt.targets[0], step=0.2, bound=2)

pen = np.array([1, 1])
lambda_a = 0.001

tic()
cg.fill_preferred_schedules(Tf=1, r_pen=pen, lambda_a=lambda_a, heuristic=False)
toc()
sch0_no_heur = []
sch1_no_heur = []
for key in list(cg.G.keys()):
    if cg.G[key].preferred_scheduling == 0:
        sch0_no_heur.append(key)
    else:
        sch1_no_heur.append(key)

tic()
cg.fill_preferred_schedules(Tf=1, r_pen=pen, lambda_a=lambda_a, heuristic=True)
toc()
sch0_heur = []
sch1_heur = []
for key in list(cg.G.keys()):
    if cg.G[key].preferred_scheduling == 0:
        sch0_heur.append(key)
    else:
        sch1_heur.append(key)


cg.save_graph(folder='graphs')

print("END")