import numpy as np
from matplotlib import pyplot as plt
from Haldane_chain_qubit import *

L = 10  # each chain length
K1 = 20  # number of states in the Sz_tot=0 of one chain
K2 = 20  # number of states to be found in the coupled pair
NJ = 20  # number of Jab points

Pair = TwoChains(L)
Single = SingleChain(L)
E = Single.eigen_system(vectors=False)
Gamma = E[0] - E[1]
Delta = E[1] - E[2]
print(Gamma / Delta)
B_ab = Gamma / 2 + Delta
J_max = Gamma / 2 - Delta
J_ab = np.linspace(0, 1 * J_max, NJ)

spect0 = Pair.uncoupled_spect(K1)
spect0 = spect0[spect0[:, 1] == 0]
spect0 = np.asarray([spect0[:, 0], spect0[:, 0] + 0.5 * B_ab * spect0[:, 2]])
print(spect0.shape)

E = np.zeros((NJ, K2))
for i in range(NJ):
    print(i)
    E[i, :], V = Pair.eigen_system_eff(bab=B_ab, jab=J_ab[i], k1=K1, k2=K2)

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
# plotting
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 8))
fig.subplots_adjust(wspace=0, right=0.96, top=0.96)

ax1.plot([0, B_ab], spect0)
ax1.set_xlim(0, B_ab)
ax1.set_ylim(spect0.min() - J_max, spect0.min() + 2 * B_ab)
ax1.set_xticks(np.arange(0, 0.5, 0.1))
ax1.tick_params(labelsize=13)
ax1.set_ylabel('$E / J$', fontsize=14)
ax1.set_xlabel('$B_{AB} / J$', fontsize=14)

ax2.plot(J_ab, E)
ax2.set_xlim(0, J_ab.max())
ax2.set_ylim(spect0.min() - J_max, spect0.min() + 2 * B_ab)
ax2.tick_params(labelsize=13)
ax2.set_xlabel("$J' / J$", fontsize=14)

# plt.tight_layout()
plt.show()
