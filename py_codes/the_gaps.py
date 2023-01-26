import numpy as np
from matplotlib import pyplot as plt
from Haldane_chain_qubit import *

L = 17  # maximum length
K = 6  # number of eigenvalues found for each chain

try:
    with open('data/gaps.txt') as f:
        d = len(f.readlines())
        if L > d:
            with open('data/gaps.txt', 'a') as fa:
                print('adding new data to the file...')
                E = np.zeros((L - d, K))
                start = time.time()
                for i in range(d + 1, L + 1):
                    Chain = SingleChain(i)
                    E[i - d - 1, :] = np.sort(Chain.eigen_system(k=K, vectors=False))
                    fa.write(str(i) + '\t' +
                             str(E[i - d - 1, 1] - E[i - d - 1, 0]) + '\t' +
                             str(E[i - d - 1, 2] - E[i - d - 1, 1]) + '\n')
                    print('chain #', i, 'done')
                end = time.time()
                print('computation took', end - start, 'seconds')
        f.seek(0)
        # d = len(f.readlines())
        # print(d)
        # f.seek(0)
        data = f.read()
        data = data.split()[3:]
        data = np.array(list(map(float, data))).reshape((len(data) // 3, 3))
        Length = data[:, 0].astype(int)
        Delta = data[:, 1]
        Gamma = data[:, 2]

except IOError:
    start = time.time()
    print('saved data is not available')
    print('calculating the gaps now...')
    with open('data/gaps.txt', 'w') as f:
        f.write('#' + '\t' + 'Delta' + 3 * '\t' + 'Gamma' + '\n')
        f.write('2' + '\t' + '1' + 3 * '\t' + '2' + '\n')
        E = np.zeros((L - 1, K))
        E[0, :3] = [-2, -1, 1]
        for i in range(3, L + 1):
            Chain = SingleChain(i)
            E[i - 2, :] = np.sort(Chain.eigen_system(k=K, vectors=False))
            f.write(str(i) + '\t' +
                    str(E[i - 2, 1] - E[i - 2, 0]) + '\t' +
                    str(E[i - 2, 2] - E[i - 2, 1]) + '\n')
            print('chain #', i, 'done')
    end = time.time()
    Length = range(2, L + 1)
    Delta = E[:, 1] - E[:, 0]
    Gamma = E[:, 2] - E[:, 1]
    print('computation took', end - start, 'seconds')

''' Read MPS results by Dan '''
with open('data/MPS_Gaps_Dan.dat') as f:
    data = f.read()
    data = data.split()[4:]
    data = np.array(list(map(float, data))).reshape((len(data) // 4, 4))
    Length_mps = data[:, 0].astype(int)
    E_S = data[:, 1]
    E_T = data[:, 2]
    E_Q = data[:, 3]
    Delta_mps = E_T - E_S
    Gamma_mps = E_Q - E_T

print(Gamma/Delta)

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
# plotting
plt.plot(Length, Delta, 'o', label='$\Delta$ with ED')
plt.plot(Length_mps, Delta_mps, 'x', label='$\Delta$ with MPS')
plt.plot(Length, Gamma, 'o', label='$\Gamma$ with ED')
plt.plot(Length_mps, Gamma_mps, 'x', label='$\Gamma$ with MPS')
plt.axhline(y=0.4, ls=':', color='grey')
plt.yscale('log')
plt.xticks(range(2, 40, 2), fontsize=12)
plt.yticks([1e-3, 1e-2, 1e-1, 1, 2], [1e-3, 1e-2, 1e-1, 1, 2], fontsize=12)
plt.xlabel('chain length', fontsize=16)
plt.ylabel('$E / J$', fontsize=16)
plt.text(18, 0.25, '$E = 0.4 J$', fontsize=16, color='grey')
plt.legend(fontsize=15, frameon=False)
plt.tight_layout()

plt.figure()
plt.stem(Length_mps, Gamma_mps / 2 - Delta_mps, '.')
plt.xlabel('chain length', fontsize=16)
plt.ylabel('$(\Gamma - 2 \Delta) / 2J$', fontsize=16)
plt.ylim(0.16)
plt.xticks(range(2, 40, 2), fontsize=12)
plt.tick_params(labelsize=12)

plt.show()
