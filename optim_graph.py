from source.covariance_graph import CovarianceGraph
from utils import load_object
import numpy as np
from source.target_model import MultiTarget

np.set_printoptions(linewidth=200)

#cg = load_object('graph11_12_2021-11_58_04')
#cg = load_object('graph11_12_2021-13_05_48')
#cg = load_object('graph11_16_2021-16_31_09')

pcov = np.array([0.1])
deltas = [0.4, 0.1]
mcovs = [np.array([0.1]), np.array([1])]

mt = MultiTarget()
mt.add_target(number=1, order=2, ws_dim=1, pcov=pcov)
mt.set_random_x0()
mt.set_latency(deltas, mcovs)

cg = CovarianceGraph(mt.targets[0], step=0.1, bound=1)

P0 = np.array([[0.9, 0.1], [0.1, 0.9]])
P0, q0 = cg.quantize(P0)

pen = np.array([1, 1])
lambda_a = 0.001
p, q, Jdp = cg.dp_search(q0=q0, Tf=1, r_pen=pen, lambda_a=lambda_a)

#print((p, J))

Jpg = cg.cost_of_schedule(schedule=p, q0=q0, Tf=1, r_pen=pen, lambda_a=lambda_a)
Jprg = cg.real_cost_of_schedule(schedule=p, q0=q0, Tf=1, r_pen=pen, lambda_a=lambda_a)
#print((Jpg, Jprg))

print((Jdp, Jpg))
costs = []
for i in range(0, 1000):
    p = list(np.random.randint(low=0, high=2, size=12))
    Jp = cg.cost_of_schedule(schedule=p, q0=q0, Tf=1, r_pen=pen, lambda_a=lambda_a)
    Jpr = cg.real_cost_of_schedule(schedule=p, q0=q0, Tf=1, r_pen=pen, lambda_a=lambda_a)
    #print((p[0:10], Jp, Jpr))
    costs.append(Jpr)

costs = np.array(costs)
print((Jprg, costs.min(), costs.mean()))
