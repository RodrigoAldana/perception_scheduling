from covariance_graph import CovarianceGraph
from utils import save_object
import numpy as np
from target_model import MultiTarget


np.set_printoptions(linewidth=200)

pcov = np.array([1])
deltas = [0.4, 0.1]
mcovs = [np.array([0.5]), np.array([0.005])]

mt = MultiTarget()
mt.add_target(number=1, order=2, ws_dim=1, pcov=pcov)
mt.set_latency(deltas, mcovs)
cg = CovarianceGraph(mt.targets[0], step=0.2, bound=2, exhaustive=False, n_samples=100)

pen = np.array([1, 1])
lambda_a = 0.5
N_schedules = 100
Tf = 1
N_points = 1000

Jq_data, Jreal_data, Jrandoms_data, J0_data, J1_data = [], [], [], [], []
for i_point in range(0, N_points):
    print(('step: ', i_point+1, ' out of: ', N_points))
    P0= 2 * (np.random.rand(2, 2) - 0.5)
    P0 = cg.bound * (P0.T @ P0)
    P0, q0 = cg.closest(P0)

    p, q, Jq = cg.dp_search_2d(q0=(q0, q0), Tf=Tf, r_pen=pen, lambda_a=lambda_a, silent=False)
    Jq_data.append(Jq)

    Jreal = cg.real_cost_of_schedule_2d(schedule=p, q0=(q0, q0), Tf=Tf, r_pen=pen, lambda_a=lambda_a)
    Jreal_data.append(Jreal)

    Jrandoms = []
    for i in range(0, N_schedules):
        p = list(np.random.randint(low=0, high=2, size=12))
        Jrandom = cg.real_cost_of_schedule_2d(schedule=p, q0=(q0, q0), Tf=Tf, r_pen=pen, lambda_a=lambda_a)
        Jrandoms.append(Jrandom)
    Jrandom = np.array(Jrandom)
    Jrandoms_data.append(Jrandom.mean())

    J0 = cg.real_cost_of_schedule_2d(schedule=[0], q0=(q0, q0), Tf=Tf, r_pen=pen, lambda_a=lambda_a)
    J0_data.append(J0)

    J1 = cg.real_cost_of_schedule_2d(schedule=[1], q0=(q0, q0), Tf=Tf, r_pen=pen, lambda_a=lambda_a)
    J1_data.append(J1)

    print(('Quantized cost', Jq))
    print(('Real cost', Jreal))
    print(('Average random cost', Jrandom))
    print(('Cost of 0', J0))
    print(('Cost of 1', J1))


save_object(Jq_data, name='Jq', folder='data', include_date=False)
save_object(Jreal_data, name='Jreal', folder='data', include_date=False)
save_object(Jrandoms_data, name='Jrandoms', folder='data', include_date=False)
save_object(J0_data, name='J0', folder='data', include_date=False)
save_object(J1_data, name='J1', folder='data', include_date=False)
