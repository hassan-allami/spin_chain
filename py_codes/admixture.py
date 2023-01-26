from Haldane_chain_qubit import *
import numpy as np
from matplotlib import pyplot as plt

# comparing the admixture between length L1 and L2
L1 = 8
L2 = 14
K = 6  # the number of states found
N = 20  # the number of B1 points

with open('data/gaps.txt') as f:
    data1 = f.readlines()[L1 - 1].split()[1:]
    f.seek(0)
    data2 = f.readlines()[L2 - 1].split()[1:]
Delta1, Gamma1 = list(map(float, data1))
Delta2, Gamma2 = list(map(float, data2))
B1_max = 1.1 * Delta1
print('Delta1 =', Delta1)
print('Delta2 =', Delta2)

try:
    A1 = np.zeros((2, 2, N))
    A2 = np.zeros((2, 2, N))
    B1 = np.loadtxt('data/B1_L' + str(L1) + '-' + str(L2))
    eps1 = np.loadtxt('data/admix1_L' + str(L1) + '-' + str(L2))
    eps2 = np.loadtxt('data/admix2_L' + str(L1) + '-' + str(L2))
    A1[:, 0, :] = np.loadtxt('data/a0_1_L' + str(L1) + '-' + str(L2))
    A1[:, 1, :] = np.loadtxt('data/a1_1_L' + str(L1) + '-' + str(L2))
    A2[:, 0, :] = np.loadtxt('data/a0_2_L' + str(L1) + '-' + str(L2))
    A2[:, 1, :] = np.loadtxt('data/a1_2_L' + str(L1) + '-' + str(L2))
except IOError:
    B1 = np.linspace(0, B1_max, N)
    eps1 = np.zeros((N, 2))
    eps2 = np.zeros((N, 2))
    A1 = np.zeros((2, 2, N))
    A2 = np.zeros((2, 2, N))

    Chain1 = SingleChain(L1)
    Chain2 = SingleChain(L2)
    start = time.time()
    print('saved data is not available')
    print('computing...')
    for i in range(N):
        eps1[i] = Chain1.admix(B1[i])
        eps2[i] = Chain2.admix(B1[i])
        E01, V01 = Chain1.eigen_system()
        E1, V1 = Chain1.eigen_system(b=B1[i])
        E02, V02 = Chain2.eigen_system()
        E2, V2 = Chain2.eigen_system(b=B1[i])
        A1[:, :, i] = (V01[:, :2].T @ V1[:, :2]) ** 2
        A2[:, :, i] = (V02[:, :2].T @ V2[:, :2]) ** 2
        print(i)
    end = time.time()
    print('computation took', end - start, 'seconds')
    if max(L1, L2) > 10:
        np.savetxt('data/B1_L' + str(L1) + '-' + str(L2), B1)
        np.savetxt('data/admix1_L' + str(L1) + '-' + str(L2), eps1)
        np.savetxt('data/admix2_L' + str(L1) + '-' + str(L2), eps2)
        np.savetxt('data/a0_1_L' + str(L1) + '-' + str(L2), A1[:, 0, :])
        np.savetxt('data/a1_1_L' + str(L1) + '-' + str(L2), A1[:, 1, :])
        np.savetxt('data/a0_2_L' + str(L1) + '-' + str(L2), A2[:, 0, :])
        np.savetxt('data/a1_2_L' + str(L1) + '-' + str(L2), A2[:, 1, :])

# print(sum(A1[1, :, :]) + eps1[:, 1])
# print(sum(A1[0, :, :]) + eps1[:, 0])
# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
# plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 10), sharex=True)
fig.subplots_adjust(hspace=0, left=0.155, right=0.95, bottom=0.1, top=0.95)

ax1.set_title('solid: $N =$ '+str(L2)+', dashed: $N =$ '+str(L1), fontsize=15)
ax1.plot(B1, A1[0, :, :].T, '--')
ax1.set_prop_cycle(None)
ax1.plot(B1, A2[0, :, :].T, '-', label=['$|a_{00}|^2$', '$|a_{01}|^2$'])
ax1.set_ylim(0, 1.2)
ax1.set_xlim(0, B1_max)
ax1.set_xlabel('$B_1 / J$', fontsize=15)
ax1.tick_params(labelsize=14)
ax1.axvline(x=Delta1, ls='--', color='grey')
ax1.axvline(x=Delta2, ls='-', color='grey')
# ax1.axvline(x=Gamma1/5, ls='--', color='C2')
# ax1.axvline(x=Gamma2/5, ls='-', color='C2')
ax1.legend(fontsize=14, frameon=False)

ax2.plot(B1, A1[1, :, :].T, '--')
ax2.set_prop_cycle(None)
ax2.plot(B1, A2[1, :, :].T, '-', label=['$|a_{10}|^2$', '$|a_{11}|^2$'])
ax2.set_ylim(0, 1.2)
ax2.set_xlim(0, B1_max)
# ax2.set_ylabel('$|a_{1j}|^2$', fontsize=15)
ax2.tick_params(labelsize=14)
ax2.axvline(x=Delta1, ls='--', color='grey')
ax2.axvline(x=Delta2, ls='-', color='grey')
# ax2.axvline(x=Gamma1/5, ls='--', color='C2')
# ax2.axvline(x=Gamma2/5, ls='-', color='C2')
ax2.legend(fontsize=14, frameon=False)

ax3.plot(B1, eps1, '--')
ax3.set_prop_cycle(None)
ax3.plot(B1, eps2, '-', label=[r'$\varepsilon_0$', r'$\varepsilon_1$'])
ax3.set_ylim(0)
ax3.set_xlim(0, B1_max)
ax3.set_xlabel('$B_1 / J$', fontsize=15)
ax3.set_ylabel('Admixture', fontsize=15)
ax3.set_yticks([0, 0.01])
ax3.tick_params(labelsize=14)
ax3.axvline(x=Delta1, ls='--', color='grey')
ax3.axvline(x=Delta2, ls='-', color='grey')
# ax3.axvline(x=Gamma1/5, ls='--', color='C2')
# ax3.axvline(x=Gamma2/5, ls='-', color='C2')
ax3.legend(fontsize=14, frameon=False)

plt.show()
