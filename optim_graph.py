from source.covariance_graph import CovarianceGraph
from utils import load_object
import numpy as np

np.set_printoptions(linewidth=200)

#cg = load_object('graph11_12_2021-11_58_04')
#cg = load_object('graph11_12_2021-13_05_48')
cg = load_object('graph11_15_2021-17_32_38')

P0 = np.array([[0.8, 0], [0, 0.8]])
P0, q0 = cg.quantize(P0)

p, J = cg.dp_search(q0=q0, Tf=1, r_pen=np.array([1, 1]), lambda_a=1)

print((p, J))

Jp = cg.cost_of_schedule(schedule=p, q0=q0, Tf=1, r_pen=np.array([1, 1]), lambda_a=1)
print(Jp)

p = list(np.random.randint(low=0, high=2, size=100))
Jp = cg.cost_of_schedule(schedule=p, q0=q0, Tf=1, r_pen=np.array([1, 1]), lambda_a=1)
print(Jp)

p = list(np.random.randint(low=0, high=2, size=100))
Jp = cg.cost_of_schedule(schedule=p, q0=q0, Tf=1, r_pen=np.array([1, 1]), lambda_a=1)
print(Jp)

p = list(np.random.randint(low=0, high=2, size=100))
Jp = cg.cost_of_schedule(schedule=p, q0=q0, Tf=1, r_pen=np.array([1, 1]), lambda_a=1)
print(Jp)

print('END')
