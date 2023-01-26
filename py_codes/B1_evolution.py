from Haldane_chain_qubit import *
import numpy as np
from matplotlib import pyplot as plt

L = 8  # the chain length
K = 6  # the number of states found
M = 2  # maximum |S_tot^z| considered
N = 20  # the number of B1 points

with open('data/gaps.txt') as f:
    data = f.readlines()[L - 1].split()[1:]
Delta, Gamma = list(map(float, data))
B_max = (Gamma + Delta) / 3
B1 = np.linspace(0, B_max / 2, N)
E = np.zeros((N, K, 2*M + 1))
EF = np.zeros((N, 9))

if L < 10:
    Chain = SingleChain(L)
    start = time.time()
    for m in range(M + 1):
        for i in range(N):
            if m == 0:
                E[i, :, m] = np.sort(Chain.eigen_system(m, B1[i], k=K, vectors=False))
            else:
                E[i, :, m] = np.sort(Chain.eigen_system(m, B1[i], k=K, vectors=False))
                E[i, :, -m] = np.sort(Chain.eigen_system(-m, B1[i], k=K, vectors=False))
        print('|Sz| =', m, 'done')
    # for i in range(N):
    #     EF[i, :] = np.sort(Chain.eigen_system_full(B_max, B1[i], vectors=False))
    # print(EF.shape)
    end = time.time()
    print('computation took', end - start, 'seconds')
else:
    try:
        B1 = np.loadtxt('data/B1_L' + str(L))
        for m in range(-M, M+1):
            E[:, :, m] = np.loadtxt('data/B1_evol_L' + str(L) + '_m' + str(m))
    except IOError:
        start = time.time()
        print('saved data is not available')
        print('computing...')
        Chain = SingleChain(L)
        for m in range(M + 1):
            for i in range(N):
                if m == 0:
                    E[i, :, m] = np.sort(Chain.eigen_system(m, B1[i], k=K, vectors=False))
                else:
                    E[i, :, m] = np.sort(Chain.eigen_system(m, B1[i], k=K, vectors=False))
                    E[i, :, -m] = np.sort(Chain.eigen_system(-m, B1[i], k=K, vectors=False))
            print('|Sz| =', m, 'done')
        end = time.time()
        print('computation for L =', L, 'took', end - start, 'seconds')
        np.savetxt('data/B1_L' + str(L), B1)
        for m in range(-M, M+1):
            np.savetxt('data/B1_evol_L' + str(L) + '_m' + str(m), E[:, :, m])

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
# plotting
plt.plot(B1, E[:, :, 0], lw=2)
plt.gca().set_prop_cycle(None)
plt.plot(B1, E[:, :, 1] + 1*B_max, '--')
plt.gca().set_prop_cycle(None)
plt.plot(B1, E[:, :, -1] - 1*B_max, '--')
plt.gca().set_prop_cycle(None)
plt.plot(B1, E[:, :, 2] + 2*B_max, '--')
plt.gca().set_prop_cycle(None)
plt.plot(B1, E[:, :, -2] - 2*B_max, '--')
plt.xlim(0, B_max/2)
plt.ylim(min(E[:, 0, 0]) - B_max, min(E[:, 5, 0]))
plt.tick_params(labelsize=14)
plt.xlabel('$B_1 / J$', fontsize=15)
plt.ylabel('$E / J$', fontsize=15)
plt.title('chain length = ' + str(L), fontsize=16)
plt.tight_layout()

# plt.plot(B1, EF, '.')

plt.figure()
plt.plot(B1, E[:, :, 0], lw=2)
plt.gca().set_prop_cycle(None)
plt.plot(B1, E[:, :, 1] + 0*B_max, '--')
plt.gca().set_prop_cycle(None)
plt.plot(B1, E[:, :, -1] - 0*B_max, '--')
plt.gca().set_prop_cycle(None)
plt.plot(B1, E[:, :, 2] + 0*B_max, '--')
plt.gca().set_prop_cycle(None)
plt.plot(B1, E[:, :, -2] - 0*B_max, '--')
plt.xlim(0, B_max/2)
plt.ylim(min(E[:, 0, 0]) - B_max, min(E[:, 5, 0]))
plt.tick_params(labelsize=14)
plt.xlabel('$B_1 / J$', fontsize=15)
plt.ylabel('$E / J$', fontsize=15)
plt.title('chain length = ' + str(L), fontsize=16)
plt.tight_layout()

plt.show()
