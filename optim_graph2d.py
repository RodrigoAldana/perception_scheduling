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

cg = CovarianceGraph(mt.targets[0], step=0.2, bound=2, exhaustive=False, n_samples=50)

P0 = np.array([[0.9, 0.1], [0.1, 0.9]])
P0, q0 = cg.closest(P0)

pen = np.array([1, 1])
lambda_a = 0.1

tic()
p, q, J = cg.dp_search(q0=q0, Tf=1, r_pen=pen, lambda_a=lambda_a, silent=False)
toc()

tic()
p, q, J = cg.dp_search_2d(q0=(q0, q0), Tf=1, r_pen=pen, lambda_a=lambda_a, silent=False)
toc()

print((p, J))

Jpg = cg.cost_of_schedule_2d(schedule=p, q0=(q0, q0), Tf=1, r_pen=pen, lambda_a=lambda_a)
Jprg = cg.real_cost_of_schedule_2d(schedule=p, q0=(q0, q0), Tf=1, r_pen=pen, lambda_a=lambda_a)
print((Jpg, Jprg))

costs = []
for i in range(0, 10):
    p = list(np.random.randint(low=0, high=2, size=12))
    Jp = cg.cost_of_schedule_2d(schedule=p, q0=(q0, q0), Tf=1, r_pen=pen, lambda_a=lambda_a)
    Jpr = cg.real_cost_of_schedule_2d(schedule=p, q0=(q0, q0), Tf=1, r_pen=pen, lambda_a=lambda_a)
    costs.append(Jpr)

costs = np.array(costs)
print((Jprg, costs.min(), costs.mean()))


