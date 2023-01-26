import time

from Haldane_chain_qubit import *
import numpy as np
from scipy.sparse import coo_matrix, bmat, diags, hstack, linalg
from matplotlib import pyplot as plt

L = 10  # chains length
K = 5  # number of states in the S_tot^z=0 subspace of each chain
Pair = TwoChains(L)
chain = SingleChain(L)
print(chain.levels_s_tot(k=7)[1])

start = time.time()
spect = Pair.uncoupled_spect(k=K)
end = time.time()
print('it took', end - start, 'seconds')

spect0 = spect[spect[:, 1] == 0]
spect1 = spect[abs(spect[:, 1]) == 1]
print(spect.shape)
print(spect0.shape)
print(spect1.shape)
E0 = sorted(spect0[:, 0])
Delta = E0[1] - E0[0]
Gamma = E0[6] - E0[5] + Delta

B_max = 5
B_avg = 0.2
Spect = np.asarray([spect[:, 0], spect[:, 0] + B_max / 2 * spect[:, 2]] + B_avg * spect[:, 1])
Spect0 = np.asarray([spect0[:, 0], spect0[:, 0] + B_max/2 * spect0[:, 2]])
Spect1 = np.asarray([spect1[:, 0], spect1[:, 0] + B_max/2 * spect1[:, 2]] + B_avg * spect1[:, 1])
# print('test', spect0[:, 2])

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
# plotting
plt.plot([0, B_max], Spect0)
# plt.xlim(0, B_max)
plt.xlim(0, Gamma / 2 + Delta)
plt.ylim(E0[0] - Gamma / 2 + Delta, E0[10] + 0.1)
plt.xlabel('$B_{AB}/J$', fontsize=14)
plt.ylabel('$E/J$', fontsize=14)
plt.tick_params(labelsize=14)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, Gamma / 2 + Delta], [0, 0.1, 0.2, 0.3, 0.4, r'$\frac{\Gamma + 2 \Delta}{2J}$'])
# plt.plot([0, B_max], Spect1, '--')
# plt.plot([0, B_max], Spect, ':')


plt.tight_layout()
plt.show()
