from utils import load_object
import numpy as np
import matplotlib.pyplot as plt

Jq = np.array(load_object('data/Jq'))
Jreal = np.array(load_object('data/Jreal'))
Jrandoms = np.array(load_object('data/Jrandoms'))
J0 = np.array(load_object('data/J0'))
J1 = np.array(load_object('data/J1'))
print(('Real cost', Jreal.mean()))
print(('Cost of 0', J0.mean()))
print(('Cost of 1', J1.mean()))
print(('Quantized cost', Jq.mean()))
print(('Average random cost', Jrandoms.mean()))

fig, ax = plt.subplots()
alpha = 0.7
bins = 30

ax.hist(Jq, bins=bins, label='Jq', alpha=alpha)
ax.hist(J0, bins=bins, label='J0', alpha=alpha)
ax.hist(J1, bins=bins, label='J1', alpha=alpha)

x = False
if x:
    ax.hist(Jreal, bins=bins, label='Jreal', alpha=alpha)
    ax.hist(Jrandoms, bins=bins, label='Jrandoms', alpha=alpha)

legend = ax.legend(loc='upper right')
plt.show()
