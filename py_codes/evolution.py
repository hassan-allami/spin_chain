from Haldane_chain_qubit import *
import numpy as np
from matplotlib import pyplot as plt

# comparing the admixture between length L1 and L2
L1 = 8
L2 = 14
# print(hilbert_space_size(0, L1))
# print(hilbert_space_size(0, L2))
N = 40  # the number of time steps

with open('data/gaps.txt') as f:
    data1 = f.readlines()[L1 - 1].split()[1:]
    f.seek(0)
    data2 = f.readlines()[L2 - 1].split()[1:]
Delta1, Gamma1 = list(map(float, data1))
Delta2, Gamma2 = list(map(float, data2))

try:
    print('loading from saved results')
    t1 = np.loadtxt('data/evol_t1_' + str(L1) + '-' + str(L2) + '_B1_Delta')
    A0_1_2 = np.loadtxt('data/evol_A0_1_2_' + str(L1) + '-' + str(L2) + '_B1_Delta')
    A1_1_2 = np.loadtxt('data/evol_A1_1_2_' + str(L1) + '-' + str(L2) + '_B1_Delta')
    A0_1_10 = np.loadtxt('data/evol_A0_1_10_' + str(L1) + '-' + str(L2) + '_B1_Delta')
    A1_1_10 = np.loadtxt('data/evol_A1_1_10_' + str(L1) + '-' + str(L2) + '_B1_Delta')
    A0_1_100 = np.loadtxt('data/evol_A0_1_100_' + str(L1) + '-' + str(L2) + '_B1_Delta')
    A1_1_100 = np.loadtxt('data/evol_A1_1_100_' + str(L1) + '-' + str(L2) + '_B1_Delta')

    t2 = np.loadtxt('data/evol_t2_' + str(L1) + '-' + str(L2) + '_B1_Delta')
    A0_2_2 = np.loadtxt('data/evol_A0_2_2_' + str(L1) + '-' + str(L2) + '_B1_Delta')
    A1_2_2 = np.loadtxt('data/evol_A1_2_2_' + str(L1) + '-' + str(L2) + '_B1_Delta')
    A0_2_10 = np.loadtxt('data/evol_A0_2_10_' + str(L1) + '-' + str(L2) + '_B1_Delta')
    A1_2_10 = np.loadtxt('data/evol_A1_2_10_' + str(L1) + '-' + str(L2) + '_B1_Delta')
    A0_2_100 = np.loadtxt('data/evol_A0_2_100_' + str(L1) + '-' + str(L2) + '_B1_Delta')
    A1_2_100 = np.loadtxt('data/evol_A1_2_100_' + str(L1) + '-' + str(L2) + '_B1_Delta')

except IOError:
    Chain1 = SingleChain(L1)
    t1 = np.linspace(0, 3 / Delta1, N)
    A0_1_2 = np.zeros((N, 2))
    A1_1_2 = np.zeros((N, 2))
    A0_1_10 = np.zeros((N, 2))
    A1_1_10 = np.zeros((N, 2))
    A0_1_100 = np.zeros((N, 2))
    A1_1_100 = np.zeros((N, 2))

    Chain2 = SingleChain(L2)
    t2 = np.linspace(0, 3 / Delta2, N)
    A0_2_2 = np.zeros((N, 2))
    A1_2_2 = np.zeros((N, 2))
    A0_2_10 = np.zeros((N, 2))
    A1_2_10 = np.zeros((N, 2))
    A0_2_100 = np.zeros((N, 2))
    A1_2_100 = np.zeros((N, 2))

    start = time.time()
    print('saved data is not available')
    print('computing...')
    for i in range(N):
        U1_2 = Chain1.unitary_approx(Delta1, t1[i], d=2)
        U1_10 = Chain1.unitary_approx(Delta1, t1[i], d=10)
        U1_100 = Chain1.unitary_approx(Delta1, t1[i], d=100)
        A0_1_2[i, :] = np.abs([U1_2[0, 0], U1_2[1, 0]]) ** 2
        A1_1_2[i, :] = np.abs([U1_2[0, 1], U1_2[1, 1]]) ** 2
        A0_1_10[i, :] = np.abs([U1_10[0, 0], U1_10[1, 0]]) ** 2
        A1_1_10[i, :] = np.abs([U1_10[0, 1], U1_10[1, 1]]) ** 2
        A0_1_100[i, :] = np.abs([U1_100[0, 0], U1_100[1, 0]]) ** 2
        A1_1_100[i, :] = np.abs([U1_100[0, 1], U1_100[1, 1]]) ** 2

        U2_2 = Chain2.unitary_approx(Delta2, t2[i], d=2)
        U2_10 = Chain2.unitary_approx(Delta2, t2[i], d=10)
        U2_100 = Chain2.unitary_approx(Delta2, t2[i], d=100)
        A0_2_2[i, :] = np.abs([U2_2[0, 0], U2_2[1, 0]]) ** 2
        A1_2_2[i, :] = np.abs([U2_2[0, 1], U2_2[1, 1]]) ** 2
        A0_2_10[i, :] = np.abs([U2_10[0, 0], U2_10[1, 0]]) ** 2
        A1_2_10[i, :] = np.abs([U2_10[0, 1], U2_10[1, 1]]) ** 2
        A0_2_100[i, :] = np.abs([U2_100[0, 0], U2_100[1, 0]]) ** 2
        A1_2_100[i, :] = np.abs([U2_100[0, 1], U2_100[1, 1]]) ** 2
        print(i)
    end = time.time()
    print('computation took', end - start, 'seconds')

    if max(L1, L2) > 10:
        np.savetxt('data/evol_t1_' + str(L1) + '-' + str(L2) + '_B1_Delta', t1)
        np.savetxt('data/evol_A0_1_2_' + str(L1) + '-' + str(L2) + '_B1_Delta', A0_1_2)
        np.savetxt('data/evol_A1_1_2_' + str(L1) + '-' + str(L2) + '_B1_Delta', A1_1_2)
        np.savetxt('data/evol_A0_1_10_' + str(L1) + '-' + str(L2) + '_B1_Delta', A0_1_10)
        np.savetxt('data/evol_A1_1_10_' + str(L1) + '-' + str(L2) + '_B1_Delta', A1_1_10)
        np.savetxt('data/evol_A0_1_100_' + str(L1) + '-' + str(L2) + '_B1_Delta', A0_1_100)
        np.savetxt('data/evol_A1_1_100_' + str(L1) + '-' + str(L2) + '_B1_Delta', A1_1_100)

        np.savetxt('data/evol_t2_' + str(L1) + '-' + str(L2) + '_B1_Delta', t2)
        np.savetxt('data/evol_A0_2_2_' + str(L1) + '-' + str(L2) + '_B1_Delta', A0_2_2)
        np.savetxt('data/evol_A1_2_2_' + str(L1) + '-' + str(L2) + '_B1_Delta', A1_2_2)
        np.savetxt('data/evol_A0_2_10_' + str(L1) + '-' + str(L2) + '_B1_Delta', A0_2_10)
        np.savetxt('data/evol_A1_2_10_' + str(L1) + '-' + str(L2) + '_B1_Delta', A1_2_10)
        np.savetxt('data/evol_A0_2_100_' + str(L1) + '-' + str(L2) + '_B1_Delta', A0_2_100)
        np.savetxt('data/evol_A1_2_100_' + str(L1) + '-' + str(L2) + '_B1_Delta', A1_2_100)

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
# plotting
fig, ax = plt.subplots(4, 2, sharex=False, figsize=(8, 7))
fig.subplots_adjust(hspace=0, wspace=0.3, left=0.1, right=0.96, bottom=0.09)
fig.suptitle(r'\textsf{number of states: 100 $\to$ solid, 10 $\to$ x, 2 $\to$ o}', fontsize=16)

ax[0, 0].set_title('chain length = '+str(L1)+',  $B_1 = \Delta$', fontsize=14)
ax[0, 0].plot(t1, A0_1_2, 'o', mfc='none')
ax[0, 0].set_prop_cycle(None)
ax[0, 0].plot(t1, A0_1_10, 'x')
ax[0, 0].set_prop_cycle(None)
ax[0, 0].plot(t1, A0_1_100, label=['$|U_{00}|^2$', '$|U_{01}|^2$'])
ax[0, 0].set_xlim(0, 3/Delta1)
ax[0, 0].set_ylim(0, 1.2)
ax[0, 0].tick_params(labelsize=13)
ax[0, 0].legend(fontsize=13, frameon=False)  # , loc='center left')

ax[1, 0].plot(t1, 1 - np.sum(A0_1_2, axis=1), 'o', mfc='none', color='C2')
ax[1, 0].plot(t1, 1 - np.sum(A0_1_10, axis=1), 'x', color='C2')
ax[1, 0].plot(t1, 1 - np.sum(A0_1_100, axis=1), color='C2')
ax[1, 0].set_ylabel(r'$\alpha_0$', fontsize=14)
ax[1, 0].set_xlim(0, 3/Delta1)
ax[1, 0].set_ylim(0)
ax[1, 0].tick_params(labelsize=13)
ax[1, 0].set_yticks([0, 0.01])

ax[2, 0].plot(t1, A1_1_2, 'o', mfc='none')
ax[2, 0].set_prop_cycle(None)
ax[2, 0].plot(t1, A1_1_10, 'x')
ax[2, 0].set_prop_cycle(None)
ax[2, 0].plot(t1, A1_1_100, label=['$|U_{10}|^2$', '$|U_{11}|^2$'])
ax[2, 0].set_xlim(0, 3/Delta1)
ax[2, 0].set_ylim(0, 1.2)
ax[2, 0].tick_params(labelsize=13)
ax[2, 0].legend(fontsize=13, frameon=False)  # , loc='upper right')

ax[3, 0].plot(t1, 1 - np.sum(A1_1_2, axis=1), 'o', mfc='none', color='C2')
ax[3, 0].plot(t1, 1 - np.sum(A1_1_10, axis=1), 'x', color='C2')
ax[3, 0].plot(t1, 1 - np.sum(A1_1_100, axis=1), color='C2')
ax[3, 0].set_ylabel(r'$\alpha_1$', fontsize=14)
ax[3, 0].set_xlabel('$Jt$', fontsize=14)
ax[3, 0].set_xlim(0, 3/Delta1)
ax[3, 0].set_ylim(0)
ax[3, 0].tick_params(labelsize=13)

ax[0, 1].set_title('chain length = '+str(L2)+',  $B_1 = \Delta$', fontsize=14)
ax[0, 1].plot(t2, A0_2_2, 'o', mfc='none')
ax[0, 1].set_prop_cycle(None)
ax[0, 1].plot(t2, A0_2_10, 'x')
ax[0, 1].set_prop_cycle(None)
ax[0, 1].plot(t2, A0_2_100, label=['$|U_{00}|^2$', '$|U_{01}|^2$'])
ax[0, 1].set_xlim(0, 3/Delta2)
ax[0, 1].set_ylim(0, 1.2)
ax[0, 1].tick_params(labelsize=13)
ax[0, 1].legend(fontsize=13, frameon=False)  # , loc='upper right')

ax[1, 1].plot(t2, 1 - np.sum(A0_2_2, axis=1), 'o', mfc='none', color='C2')
ax[1, 1].plot(t2, 1 - np.sum(A0_2_10, axis=1), 'x', color='C2')
ax[1, 1].plot(t2, 1 - np.sum(A0_2_100, axis=1), color='C2')
ax[1, 1].set_ylabel(r'$\alpha_0$', fontsize=14)
ax[1, 1].set_xlim(0, 3/Delta2)
ax[1, 1].set_ylim(0)
ax[1, 1].tick_params(labelsize=13)
ax[1, 1].set_yticks([0, 0.002])

ax[2, 1].plot(t2, A1_2_2, 'o', mfc='none')
ax[2, 1].set_prop_cycle(None)
ax[2, 1].plot(t2, A1_2_10, 'x')
ax[2, 1].set_prop_cycle(None)
ax[2, 1].plot(t2, A1_2_100, label=['$|U_{10}|^2$', '$|U_{11}|^2$'])
ax[2, 1].set_xlim(0, 3/Delta2)
ax[2, 1].set_ylim(0, 1.2)
ax[2, 1].tick_params(labelsize=13)
ax[2, 1].legend(fontsize=13, frameon=False)  # , loc='upper right')

ax[3, 1].plot(t2, 1 - np.sum(A1_2_2, axis=1), 'o', mfc='none', color='C2')
ax[3, 1].plot(t2, 1 - np.sum(A1_2_10, axis=1), 'x', color='C2')
ax[3, 1].plot(t2, 1 - np.sum(A1_2_100, axis=1), color='C2')
ax[3, 1].set_ylabel(r'$\alpha_1$', fontsize=14)
ax[3, 1].set_xlabel('$Jt$', fontsize=14)
ax[3, 1].set_xlim(0, 3/Delta2)
ax[3, 1].set_ylim(0)
ax[3, 1].tick_params(labelsize=13)

plt.show()
