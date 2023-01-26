import numpy as np
from matplotlib import pyplot as plt
from Haldane_chain_qubit import *

L = 10  # each chain length
K1 = 10  # number of states in the Sz_tot=0 of one chain
K2 = 10  # number of states to be found in the coupled pair
NB = 30  # number of Bab points

Pair = TwoChains(L)
Single = SingleChain(L)
E = Single.eigen_system(vectors=False)
Gamma = E[0] - E[1]
Delta = E[1] - E[2]
print(Gamma / Delta)

B_max = Gamma / 2 + 2 * Delta
J_ab = 0.05  # Gamma / 2 - Delta
B_ab = np.linspace(0, B_max, NB)
E_inf, V_inf = Pair.eigen_system_eff_pol(J_ab, K1, K2)
# print(E_inf)
Admix_inf = 1 - sum(V_inf[[0, 1, K1, K1 + 1], :] ** 2)
# print(np.round(Admix_inf))
Admix = np.zeros((NB, K2))
Admix2 = np.zeros((NB, K2))
E = np.zeros((NB, K2))
E2 = np.zeros((NB, K2))
for i in range(NB):
    Sys, States = Pair.eigen_system_eff(bab=B_ab[i], jab=J_ab, k1=K1, k2=K2)
    # print(States)
    V = Sys[1]
    print(i, V.shape)
    idx = compute_basis(States)
    # print(idx)
    Admix[i, :] = 1 - sum(V[idx, :] ** 2)
    E[i, :] = Sys[0]
    # print(i)
    Sys, States = Pair.eigen_system_eff(bab=B_ab[i], jab=J_ab, k1=2*K1, k2=K2)
    # print(States)
    V = Sys[1]
    # print(V.shape)
    idx = compute_basis(States)
    # print(idx)
    Admix2[i, :] = 1 - sum(V[idx, :] ** 2)
    E2[i, :] = Sys[0]

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
# plotting
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 7))
fig.subplots_adjust(hspace=0, right=0.96, left=0.18, top=0.96, bottom=0.1)

ax1.plot(B_ab, E2[:, :5], '-')
ax1.set_prop_cycle(None)
# ax1.plot(B_ab, E[:, :5], 'x')
ax1.set_xlim(0, B_max)
ax1.set_ylim(E_inf[0] - 2 * J_ab)
ax1.set_ylabel('$E / J$', fontsize=14)
ax1.tick_params(labelsize=13)
ax1.axvline(x=Delta/2, ls='--', color='grey')
ax1.axvline(x=B_max - Delta, ls='--', color='grey')
ax1.axhline(y=E_inf[0], ls=':', color='C1')
ax1.axhline(y=E_inf[1], ls=':', color='C2')
ax1.axhline(y=E_inf[2], ls=':', color='C3')
ax1.axhline(y=E_inf[3], ls=':', color='C4')

ax2.plot(B_ab, Admix[:, :5], 'x')
ax2.set_prop_cycle(None)
ax2.plot(B_ab, Admix2[:, :5], '.')
ax2.set_yscale('log')
ax2.tick_params(labelsize=13)
ax2.set_xlabel("$B_{AB} / J$", fontsize=14)
ax2.set_ylabel('admixture', fontsize=14)
ax2.axvline(x=Delta/2, ls='--', color='grey')
ax2.axvline(x=B_max - Delta, ls='--', color='grey')
ax2.axhline(y=0.01, ls='--', color='grey')
ax2.axhline(y=Admix_inf[0], ls=':', color='C1')
ax2.axhline(y=Admix_inf[1], ls=':', color='C2')
ax2.axhline(y=Admix_inf[2], ls=':', color='C3')
ax2.axhline(y=Admix_inf[3], ls=':', color='C4')

plt.show()
