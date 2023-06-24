import numpy as np
from matplotlib import pyplot as plt
from Spin1Chain import *

N_odd = 11  # length of odd chain
N_eve = 12  # length of odd chain
Odd = Spin1Chain(N_odd)
Eve = Spin1Chain(N_eve)
print('space size for Odd =', Odd.size)
print('space size for Eve =', Eve.size)

if N_odd > 9:
    try:
        print('loading...')
        HL_odd = spr.load_npz(f'data/BLBQ/HL_N_{N_odd}.npz')
        HQ_odd = spr.load_npz(f'data/BLBQ/HQ_N_{N_odd}.npz')

    except IOError:
        print('computing large matrices...')
        HL_odd = Odd.heisenberg_ham()
        HQ_odd = Odd.bq_ham()
        spr.save_npz(f'data/BLBQ/HL_N_{N_odd}.npz', HL_odd)
        spr.save_npz(f'data/BLBQ/HQ_N_{N_odd}.npz', HQ_odd)
else:
    HL_odd = Odd.heisenberg_ham()
    HQ_odd = Odd.bq_ham()

if N_eve > 9:
    try:
        print('loading...')
        HL_eve = spr.load_npz(f'data/BLBQ/HL_N_{N_eve}.npz')
        HQ_eve = spr.load_npz(f'data/BLBQ/HQ_N_{N_eve}.npz')

    except IOError:
        print('computing large matrices...')
        HL_eve = Eve.heisenberg_ham()
        HQ_eve = Eve.bq_ham()
        spr.save_npz(f'data/BLBQ/HL_N_{N_eve}.npz', HL_eve)
        spr.save_npz(f'data/BLBQ/HQ_N_{N_eve}.npz', HQ_eve)
else:
    HL_eve = Eve.heisenberg_ham()
    HQ_eve = Eve.bq_ham()

NB = 100  # number of Beta points
Beta = np.linspace(-0.91, 1.21, NB)
K = 3  # number of eigenstates
E_odd = np.zeros((NB, K))
S_odd = np.zeros_like(E_odd)
E_eve = np.zeros((NB, K))
S_eve = np.zeros_like(E_eve)
for i in range(NB):
    H_odd = HL_odd + Beta[i] * HQ_odd
    E_odd[i], V = spr.linalg.eigsh(H_odd, K, which='SA')
    S_tot = Odd.s_tot()
    S_odd[i] = (V.T @ S_tot @ V).diagonal().round()
    print(f'S_odd{i} =', S_odd[i])
    H_eve = HL_eve + Beta[i] * HQ_eve
    E_eve[i], V = spr.linalg.eigsh(H_eve, K, which='SA')
    S_tot = Eve.s_tot()
    S_eve[i] = (V.T @ S_tot @ V).diagonal().round()
    print(f'S_eve{i} =', S_eve[i])

# making list of color codes associated with S_tot
S_color_odd = [[f'C{int(S_odd[a, k])}' for a in range(NB)] for k in range(K)]
S_color_eve = [[f'C{int(S_eve[a, k])}' for a in range(NB)] for k in range(K)]

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
# plotting
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
fig.subplots_adjust(hspace=0, right=0.96, left=0.18, top=0.96, bottom=0.1)

ax1.scatter(Beta, E_odd[:, 0] - E_odd[:, 0], c=S_color_odd[0], s=5)
ax1.scatter(Beta, E_odd[:, 1] - E_odd[:, 0], c=S_color_odd[1], s=5)
ax1.scatter(Beta, E_odd[:, 2] - E_odd[:, 0], c=S_color_odd[2], s=5)
ax1.axvline(x=1/3, ls='--', color='grey')
ax1.axvline(x=0, ls='--', color='grey')
ax1.tick_params(labelsize=13)

ax2.scatter(Beta, E_eve[:, 0] - E_eve[:, 0], c=S_color_eve[0], s=5)
ax2.scatter(Beta, E_eve[:, 1] - E_eve[:, 0], c=S_color_eve[1], s=5)
ax2.scatter(Beta, E_eve[:, 2] - E_eve[:, 0], c=S_color_eve[2], s=5)
ax2.axvline(x=1/3, ls='--', color='grey')
ax2.axvline(x=0, ls='--', color='grey')
ax2.tick_params(labelsize=13)
ax2.set_xlabel(r'$\beta$', fontsize=16)

plt.show()

