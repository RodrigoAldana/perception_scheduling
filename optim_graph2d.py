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

cg = CovarianceGraph(mt.targets[0], step=0.2, bound=2, exhaustive=False, n_samples=200)

P0 = np.array([[0.9, 0.2], [0.2, 0.9]])
P0, q0 = cg.closest(P0)

pen = np.array([1, 1])
lambda_a = 0.1
N_schedules = 100
Tf = 1

p, q, J = cg.dp_search_2d(q0=(q0, q0), Tf=Tf, r_pen=pen, lambda_a=lambda_a, silent=False)

print('\n\n\n\n\n\n\n\n')
print(p)
Jpg = cg.cost_of_schedule_2d(schedule=p, q0=(q0, q0), Tf=Tf, r_pen=pen, lambda_a=lambda_a)
Jprg = cg.real_cost_of_schedule_2d(schedule=p, q0=(q0, q0), Tf=Tf, r_pen=pen, lambda_a=lambda_a)

costs = []
for i in range(0, N_schedules):
    p = list(np.random.randint(low=0, high=2, size=12))
    Jpr = cg.real_cost_of_schedule_2d(schedule=p, q0=(q0, q0), Tf=Tf, r_pen=pen, lambda_a=lambda_a)
    costs.append(Jpr)
costs = np.array(costs)

J0 = cg.real_cost_of_schedule_2d(schedule=[0], q0=(q0, q0), Tf=Tf, r_pen=pen, lambda_a=lambda_a)
J1 = cg.real_cost_of_schedule_2d(schedule=[1], q0=(q0, q0), Tf=Tf, r_pen=pen, lambda_a=lambda_a)

print(('Quantized cost', Jpg))
print(('Real cost', Jprg))
print(('Minimum random cost', costs.min()))
print(('Average random cost', costs.mean()))
print(('Cost of 0', J0))
print(('Cost of 1', J1))
