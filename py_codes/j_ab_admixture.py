import numpy as np
from matplotlib import pyplot as plt
from Haldane_chain_qubit import *

L = 10  # each chain length
K1 = 20  # number of states in the Sz_tot=0 of one chain
K2 = 10  # number of states to be found in the coupled pair
NJ = 25  # number of Jab points

Pair = TwoChains(L)
Single = SingleChain(L)
E = Single.eigen_system(vectors=False)
Gamma = E[0] - E[1]
Delta = E[1] - E[2]
print(L, Gamma / Delta)

Pair1 = TwoChains(L - 2)
Single = SingleChain(L - 2)
E = Single.eigen_system(vectors=False)
Gamma1 = E[0] - E[1]
Delta1 = E[1] - E[2]
print(L - 2, Gamma1 / Delta1)

Pair2 = TwoChains(L - 4)
Single = SingleChain(L - 4)
E = Single.eigen_system(vectors=False)
Gamma2 = E[0] - E[1]
Delta2 = E[1] - E[2]
print(L - 4, Gamma2 / Delta2)

B_ab = Gamma / 2 + Delta
B_ab1 = Gamma1 / 2 + Delta1
B_ab2 = Gamma2 / 2 + Delta2
J_ab = np.linspace(0, 0.1, NJ)
Admix_inf = np.zeros((NJ, K2))
Admix = np.zeros((NJ, K2))
Admix_inf1 = np.zeros((NJ, K2))
Admix1 = np.zeros((NJ, K2))
Admix_inf2 = np.zeros((NJ, K2))
Admix2 = np.zeros((NJ, K2))
for i in range(NJ):
    E_inf, V_inf = Pair.eigen_system_eff_pol(J_ab[i], K1, K2)
    Admix_inf[i, :] = 1 - sum(V_inf[[0, 1, K1, K1 + 1], :] ** 2)
    # print(np.round(Admix_inf))
    E_inf, V_inf = Pair1.eigen_system_eff_pol(J_ab[i], K1, K2)
    Admix_inf1[i, :] = 1 - sum(V_inf[[0, 1, K1, K1 + 1], :] ** 2)
    # print(np.round(Admix_inf1))
    E_inf, V_inf = Pair2.eigen_system_eff_pol(J_ab[i], K1, K2)
    Admix_inf2[i, :] = 1 - sum(V_inf[[0, 1, K1, K1 + 1], :] ** 2)
    # print(np.round(Admix_inf2))
    Sys, States = Pair.eigen_system_eff(bab=B_ab, jab=J_ab[i], k1=K1, k2=K2)
    V = Sys[1]
    idx = compute_basis(States)
    Admix[i, :] = 1 - sum(V[idx, :] ** 2)
    # print(np.round(Admix))
    Sys, States = Pair1.eigen_system_eff(bab=B_ab1, jab=J_ab[i], k1=K1, k2=K2)
    V = Sys[1]
    idx = compute_basis(States)
    Admix1[i, :] = 1 - sum(V[idx, :] ** 2)
    # print(np.round(Admix1))
    Sys, States = Pair2.eigen_system_eff(bab=B_ab2, jab=J_ab[i], k1=K1, k2=K2)
    V = Sys[1]
    idx = compute_basis(States)
    Admix2[i, :] = 1 - sum(V[idx, :] ** 2)
    # print(np.round(Admix1))
    print(i)

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
# plotting
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 8))
fig.subplots_adjust(hspace=0, right=0.95, left=0.14, bottom=0.1, top=0.94)
# fig.suptitle('solid: N = 10, dashed: N = 8, dotted: N = 6', fontsize=14)

ax1.plot(J_ab, 1e2 * Admix[:, 1:5],
         label=[r'$\varepsilon_{00}$',
                r'$\varepsilon_{01}$',
                r'$\varepsilon_{10}$',
                r'$\varepsilon_{11}$']
         )
ax1.set_prop_cycle(None)
ax1.plot(J_ab, 1e2 * Admix1[:, 1:5], '--')
ax1.set_prop_cycle(None)
ax1.plot(J_ab, 1e2 * Admix2[:, 1:5], ':')
ax1.set_xlim(0, 0.1)
ax1.set_ylim(0)
ax1.tick_params(labelsize=13)
ax1.set_xlabel("$J' / J$", fontsize=14)
ax1.set_ylabel('admixture [$\%$]', fontsize=14)
ax1.legend(frameon=False, fontsize=14)
ax1.text(0.035, 2.5, r'$B_{AB} = \frac{\Gamma}{2} + \Delta$', fontsize=14)
ax1.set_title('solid: N = 10, dashed: N = 8, dotted: N = 6', fontsize=14)

ax2.plot(J_ab, 1e2 * Admix_inf[:, :4],
         label=[r'$\varepsilon_{00}$',
                r'$\varepsilon_{01}$',
                r'$\varepsilon_{10}$',
                r'$\varepsilon_{11}$'])
ax2.set_prop_cycle(None)
ax2.plot(J_ab, 1e2 * Admix_inf1[:, :4], '--')
ax2.set_prop_cycle(None)
ax2.plot(J_ab, 1e2 * Admix_inf2[:, :4], ':')
ax2.set_xlim(0, 0.1)
ax2.set_ylim(0)
ax2.tick_params(labelsize=13)
ax2.set_xlabel("$J' / J$", fontsize=14)
ax2.set_ylabel('admixture [$\%$]', fontsize=14)
ax2.legend(frameon=False, fontsize=14)
ax2.text(0.037, 0.25, r'$B_{AB} \to \infty$', fontsize=14)

plt.show()
