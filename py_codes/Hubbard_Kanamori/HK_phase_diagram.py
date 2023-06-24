import matplotlib.pyplot as plt
import numpy as np

from HK_Hamiltonian import *


# finds the boundary of the region where the low energy levels
# have the angular momentum's (0, 1, 2)
# the same as spin-1 AF chain with even number of spins
# the y-axis is V/U and
# the x-axis is |t|/J where
# J = (3W - sqrt(W^2 +4Delta^2)) / 2
# is the separation of the triplet from the next state
# v, delta, and gamma are the fixed parameters
# we are sweeping x and y by changing U and W
# area determines the (x, y) range
# steps is dx and dy steps for searching for the boundary
# if scales=log then dx or dy are d(log(x)) or d(log(y))
# scales determines linear or log scale of x and y axes
def phase_diagram_boundary(length: int, v, delta, gamma,
                           area: np.ndarray(shape=(2, 2), dtype=float),
                           steps: np.ndarray(shape=(2,), dtype=float),
                           scales=['log', 'linear']):
    hk = HubbardKanamori(length=length)
    s_tot_op = hk.s_tot()
    x = area[0, 0]
    y = area[1, 1]
    b = []
    s = 6
    # moves horizontally
    while round(s) == 6 and x < area[0, 1]:
        s = 0
        # moves vertically
        while round(s) != 6 and y > area[1, 0]:
            j = 1 / x
            w = (3 * j + np.sqrt(j ** 2 + 8 * delta ** 2)) / 4
            h = hk.ham(v / y, w, v, delta=delta, gamma=gamma)
            e, p = spr.linalg.eigsh(h, k=3, which='SA')
            s = p[:, 2].T @ s_tot_op @ p[:, 2]
            y_old = float(y)
            if scales[1] == 'log':
                log_y = np.log10(y)
                log_y -= steps[1]
                y = 10 ** log_y
            else:
                y -= steps[1]
        # one step to the right
        x_old = float(x)
        if scales[0] == 'log':
            log_x = np.log10(x)
            log_x += steps[0]
            x = 10 ** log_x
        else:
            x += steps[0]
        b.append([x_old, y_old])
        print(len(b))
        # two steps back up
        if scales[1] == 'log':
            log_y += 2 * steps[1]
            y = 10 ** log_y
        else:
            y += 2 * steps[1]
    return np.asarray(b)


# finds the 3 first levels and their S_tot in the given Area
# the y-axis is V/U and
# the x-axis is |t|/J where
# J = (3W - sqrt(W^2 +4Delta^2)) / 2
# is the separation of the triplet from the next state
# v, delta, and gamma are the fixed parameters
# we are sweeping x and y by changing U and W
# area determines the (x, y) range
# scales determines linear or log scale of x and y axes
def phase_diagram(length: int, v, delta, gamma,
                  area: np.ndarray(shape=(2, 2), dtype=float),
                  resolution=100,
                  scales=['log', 'linear']):
    hk = HubbardKanamori(length=length)
    s_tot_op = hk.s_tot()
    if scales[0] == 'log':
        x = np.logspace(np.log10(area[0, 0]), np.log10(area[0, 1]), resolution)
    else:
        x = np.linspace(area[0, 0], area[0, 1], resolution)
    if scales[1] == 'log':
        y = np.logspace(np.log10(area[1, 0]), np.log10(area[1, 1]), resolution)
    else:
        y = np.linspace(area[1, 0], area[1, 1], resolution)
    e = np.zeros((resolution, resolution, 3))
    s = np.zeros_like(e)
    for a in range(resolution):
        print(a)
        for b in range(resolution):
            j = 1 / x[b]
            w = (3 * j + np.sqrt(j ** 2 + 8 * delta ** 2)) / 4
            h = hk.ham(v / y[a], w, v, delta=delta, gamma=gamma)
            e[a, b], p = spr.linalg.eigsh(h, k=3, which='SA')
            s[a, b] = (p.T @ s_tot_op @ p).diagonal().round()
    return e, s, x, y


# plotting the boundary for L = 2 for various parameters
L = 2
V = [5, 10, 15]
Delta = np.asarray([0, 1 / 2, 1])
Gamma = [0, 1]
# Xc is the W = |Gamma| line
q = 1.3
Xc = 2 / (q * 3 * Gamma[1] - np.sqrt(q ** 2 * Gamma[1] ** 2 + 4 * Delta ** 2))
Area = np.asarray([[1e-2, 10], [1e-2, 1]])
Steps = [1e-2, 1e-3]

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

# plotting the phase diagram boundary
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
fig.subplots_adjust(wspace=0, right=0.96, left=0.08, top=0.96, bottom=0.1)
markers = ['-', '--', ':']

for g in range(len(Gamma)):
    for n in range(len(V)):
        for d in range(len(Delta)):
            try:
                B = np.loadtxt(f'../data/HK/Boundary_L{L}_G{Gamma[g]}_V{V[n]}_D{Delta[d]}_log_linear_10_1.txt')
                # print(B.shape)
            except IOError:
                B = phase_diagram_boundary(L, V[n], Delta[d], Gamma[g], Area, Steps)
                np.savetxt(f'../data/HK/Boundary_L{L}_G{Gamma[g]}_V{V[n]}_D{Delta[d]}_log_linear_10_1.txt', B)
                print(B.shape)
            axes[g].plot(B[:, 0], B[:, 1], markers[d], color=f'C{n}')
    axes[g].set_xscale('log')
    # axes[g].set_yscale('log')
    axes[g].axhline(y=1, ls='--', color='C3')
    axes[g].set_ylim(1e-1, 1.1)
    axes[g].set_xlim(1e-2, 8)
    axes[g].tick_params(labelsize=13)

axes[0].set_ylabel(r'$V/U$', fontsize=16)
axes[1].axvline(x=Xc[0], ls='-', color='grey')
axes[1].axvline(x=Xc[1], ls='--', color='grey')
axes[1].axvline(x=Xc[2], ls=':', color='grey')

# plotting the gap ratio
fig, ax = plt.subplots(figsize=(6, 5))
fig.subplots_adjust(right=0.98, left=0.12)

E, S, X, Y = phase_diagram(2, V[1], Delta[0], Gamma[0], Area, resolution=200)
R = (E[:, :, 2] - E[:, :, 1]) / (E[:, :, 1] - E[:, :, 0])
R[S[:, :, 2] != 6] = np.nan

ax.contour(X, Y, R, levels=[1, 1.5, 1.9, 1.99, 2], colors=['C3'])
ax.set_xscale('log')
ax.set_ylim(1e-1, 1.1)
ax.set_xlabel(r'$|t|/W$', fontsize=16)
ax.set_ylabel(r'$V/U$', fontsize=16)
ax.tick_params(labelsize=13)

ax1 = ax.twiny()
# C = ax.pcolormesh(X, Y, R, shading='gouraud', vmin=0.8, vmax=2)
C = ax1.imshow(R, vmin=0.8, vmax=2, origin='lower', aspect='auto',
               extent=[X.min(), X.max(), Y.min(), Y.max()],
               interpolation='bilinear')
ax1.set_xticks([])

# doing strange stuff because imshow is stupid
ax2 = ax1.twiny()
ax2.contour(X, Y, R, levels=[1, 1.5, 1.9, 1.99, 2], colors=['C3'])
ax2.set_xscale('log')
ax2.set_xticks([])

# ax.text(1.2, 0.9, '$V=10|t|$ \n $\Delta=\Gamma=0$', fontsize=16)
ax.set_title('$N=2$, $V=10|t|$, $\Delta=\Gamma=0$', fontsize=16, pad=13)
# ax.set_box_aspect(1)
cbar = fig.colorbar(C)
cbar.ax.tick_params(labelsize=13)
# cbar.set_label(r'$G_2/G_1$', fontsize=16, rotation=-90, labelpad=25)
cbar.ax.set_title(r'$G_2/G_1$', fontsize=16, pad=15)

plt.show()
