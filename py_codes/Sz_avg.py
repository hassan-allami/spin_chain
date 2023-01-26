from Haldane_chain_qubit import *
import numpy as np
from matplotlib import pyplot as plt

L = 15  # the chain length
K = 6  # the number of states found

with open('data/gaps.txt') as f:
    data = f.readlines()[L - 1].split()[1:]
Delta, Gamma = list(map(float, data))
B_max = (Gamma + Delta) / 3
B0 = np.linspace(0, B_max, 2)
print('for L =', L, ', Gamma / Delta =', Gamma / Delta)

if L < 11:
    Chain = SingleChain(L)

    E0, V0 = Chain.eigen_system(m=0, k=K)
    Ep, Vp = Chain.eigen_system(m=1, k=K)
    Em, Vm = Chain.eigen_system(m=-1, k=K)

    Sp = Chain.sz_avg(1, Vp[:, 0])
    Sm = Chain.sz_avg(-1, Vm[:, 0])
    S0p = Chain.sz_avg(0, (V0[:, 0] + V0[:, 1]) / np.sqrt(2))
    S0m = Chain.sz_avg(0, (V0[:, 0] - V0[:, 1]) / np.sqrt(2))

    if L % 2 == 0:
        ES = np.array([E0[0]] * 2)
        ET = np.array([[E0[1]] * 3, [E0[1] + B_max, E0[1], E0[1] - B_max]])
        EQ = np.array([[E0[2]] * 5, [E0[2] + 2 * B_max, E0[2] + B_max, E0[2], E0[2] - B_max, E0[2] - 2 * B_max]])
    else:
        ES = np.array([E0[1]] * 2)
        ET = np.array([[E0[0]] * 3, [E0[0] + B_max, E0[0], E0[0] - B_max]])
        EQ = np.array([[E0[2]] * 5, [E0[2] + 2 * B_max, E0[2] + B_max, E0[2], E0[2] - B_max, E0[2] - 2 * B_max]])

else:
    try:
        Sp, Sm, S0p, S0m = np.loadtxt('data/Sz_avg_L' + str(L))
        E0 = np.loadtxt('data/spect0_L' + str(L))
        print('loading from saved data')
        if L % 2 == 0:
            ES = np.array([E0[0]] * 2)
            ET = np.array([[E0[1]] * 3, [E0[1] + B_max, E0[1], E0[1] - B_max]])
            EQ = np.array([[E0[2]] * 5, [E0[2] + 2 * B_max, E0[2] + B_max, E0[2], E0[2] - B_max, E0[2] - 2 * B_max]])
        else:
            ES = np.array([E0[1]] * 2)
            ET = np.array([[E0[0]] * 3, [E0[0] + B_max, E0[0], E0[0] - B_max]])
            EQ = np.array([[E0[2]] * 5, [E0[2] + 2 * B_max, E0[2] + B_max, E0[2], E0[2] - B_max, E0[2] - 2 * B_max]])

    except IOError:
        start = time.time()
        print('saved data is not available')
        print('calculating the <Sz> now...')
        Chain = SingleChain(L)

        E0, V0 = Chain.eigen_system(m=0, k=K)
        Ep, Vp = Chain.eigen_system(m=1, k=K)
        Em, Vm = Chain.eigen_system(m=-1, k=K)

        Sp = Chain.sz_avg(1, Vp[:, 0])
        Sm = Chain.sz_avg(-1, Vm[:, 0])
        S0p = Chain.sz_avg(0, (V0[:, 0] + V0[:, 1]) / np.sqrt(2))
        S0m = Chain.sz_avg(0, (V0[:, 0] - V0[:, 1]) / np.sqrt(2))

        if L % 2 == 0:
            ES = np.array([E0[0]] * 2)
            ET = np.array([[E0[1]] * 3, [E0[1] + B_max, E0[1], E0[1] - B_max]])
            EQ = np.array([[E0[2]] * 5, [E0[2] + 2 * B_max, E0[2] + B_max, E0[2], E0[2] - B_max, E0[2] - 2 * B_max]])
        else:
            ES = np.array([E0[1]] * 2)
            ET = np.array([[E0[0]] * 3, [E0[0] + B_max, E0[0], E0[0] - B_max]])
            EQ = np.array([[E0[2]] * 5, [E0[2] + 2 * B_max, E0[2] + B_max, E0[2], E0[2] - B_max, E0[2] - 2 * B_max]])
        end = time.time()
        print('computation for L =', L, 'took', end - start, 'seconds')

        np.savetxt('data/Sz_avg_L' + str(L), (Sp, Sm, S0p, S0m))
        np.savetxt('data/spect0_L' + str(L), E0)

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
# plotting
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.bar(range(1, L + 1), Sp)
ax1.axhline(y=0.5, ls='--', color='grey')
ax1.set_xticks(range(1, L + 1))
ax1.tick_params(axis='both', labelsize=14)
ax1.text((L + 1) / 2, 0.4, r'$\langle T_+ | S_i^z | T_+ \rangle$',
         horizontalalignment='center', verticalalignment='center', fontsize=18)

ax2.bar(range(1, L + 1), Sm)
ax2.axhline(y=-0.5, ls='--', color='grey')
ax2.set_xticks(range(1, L + 1))
ax2.tick_params(axis='both', labelsize=14)
ax2.text((L + 1) / 2, -0.4, r'$\langle T_- | S_i^z | T_- \rangle$',
         horizontalalignment='center', verticalalignment='center', fontsize=18)

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.bar(range(1, L + 1), S0p)
ax1.axhline(y=0.5, ls='--', color='grey')
ax1.axhline(y=-0.5, ls='--', color='grey')
ax1.set_xticks(range(1, L + 1))
ax1.tick_params(axis='both', labelsize=14)
ax1.text((L + 1) / 2, 0.4, r'$\frac{1}{2}\langle S + T_0 | S_i^z | S + T_0 \rangle$',
         horizontalalignment='center', verticalalignment='center', fontsize=18)

ax2.bar(range(1, L + 1), S0m)
ax2.axhline(y=0.5, ls='--', color='grey')
ax2.axhline(y=-0.5, ls='--', color='grey')
ax2.set_xticks(range(1, L + 1))
ax2.tick_params(axis='both', labelsize=14)
ax2.text((L + 1) / 2, -0.4, r'$\frac{1}{2}\langle S - T_0 | S_i^z | S - T_0 \rangle$',
         horizontalalignment='center', verticalalignment='center', fontsize=18)

plt.figure()
plt.plot(B0, EQ, color='C2', ls='--')
plt.plot(B0, ET, color='C1', ls='--')
plt.plot(B0, ES, color='C0', lw=2)
plt.plot(B0, ET[:, 1], color='C1', lw=2)
plt.plot(B0, EQ[:, 2], color='C2')
plt.tick_params(labelsize=14)
plt.xlim(0, B_max)
plt.xlabel('$B_0 / J$', fontsize=14)
plt.ylabel('$E / J$', fontsize=14)
plt.title('chain length = ' + str(L), fontsize=16)
plt.text(B_max / 50, ES[0] - (-1)**(L % 2) * 0.1 * (E0[2] - E0[1]), r'{\bf singlet}', color='C0',
         horizontalalignment='left', fontsize=14)
plt.text(B_max / 50, ET[0, 0] + (-1)**(L % 2) * 0.2 * (E0[2] - E0[1]), r'{\bf triplets}', color='C1',
         horizontalalignment='left', fontsize=14)
plt.text(B_max / 50, E0[2] + 0.2 * (E0[2] - E0[1]), r'{\bf quintuplets}', color='C2',
         verticalalignment='bottom', horizontalalignment='left', fontsize=14)
plt.tight_layout()
plt.show()
