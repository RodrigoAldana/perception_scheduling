from utils import load_object
import numpy as np
import matplotlib.pyplot as plt

folder = 'data/'

Jq = np.array(load_object(folder+'Jq'))
Jreal = np.array(load_object(folder+'Jreal'))
Jrandoms = np.array(load_object(folder+'Jrandoms'))
Jrandoms_min = np.array(load_object(folder+'Jrandoms_min'))
J0 = np.array(load_object(folder+'J0'))
J1 = np.array(load_object(folder+'J1'))

print(('Real cost', Jreal.mean()))
print(('Cost of 0', J0.mean()))
print(('Cost of 1', J1.mean()))
print(('Quantized cost', Jq.mean()))
print(('Average random cost', Jrandoms.mean()))

fig, ax = plt.subplots()
alpha = 0.7
bins = 15
absolute = True
if absolute:
    ax.hist(Jq, bins=bins, label='Jq', alpha=alpha)
    ax.hist(J0, bins=bins, label='J0', alpha=alpha)
    ax.hist(J1, bins=bins, label='J1', alpha=alpha)
    ax.hist(Jrandoms, bins=bins, label='Jrandoms-Jq', alpha=alpha)
    ax.hist(Jrandoms_min, bins=bins, label='Jrandoms_min-Jq', alpha=alpha)
else:
    ax.hist(J0-Jq, bins=bins, label='J0-Jq', alpha=alpha)
    ax.hist(J1-Jq, bins=bins, label='J1-Jq', alpha=alpha)
    ax.hist(Jrandoms-Jq, bins=bins, label='Jrandoms-Jq', alpha=alpha)
    ax.hist(Jrandoms_min-Jq, bins=bins, label='Jrandoms_min-Jq', alpha=alpha)
legend = ax.legend(loc='upper right')
plt.show()


