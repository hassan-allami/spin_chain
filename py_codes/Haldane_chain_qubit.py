import numpy as np
from scipy.sparse import coo_matrix, bmat, diags, block_diag, hstack, vstack, linalg, kron, kronsum
from scipy.special import comb
import time
from functools import lru_cache
from matplotlib import pyplot as plt
from itertools import product
import collections
import gc


# gives the size of the Hilbert space for a chain of length n and total S^z=m
@lru_cache  # this memoize
def hilbert_space_size(m, n):
    s = sum((-1) ** i * comb(n, i) * comb(2 * n - 3 * i + abs(m) - 1, n - 1) for i in range((n + abs(m)) // 3 + 1))
    return int(s)


# class of all functions for a single chain
class SingleChain:

    def __init__(self, length):
        self.length = length
        # self.configs(0)
        # self.rec_chain_hamiltonian_szm(0)
        # self.local_field(0)

    # gives the list of configurations for a chain of length n and total S^z=m
    def configs(self, m):
        sz_values = np.array([-1, 0, 1])
        c = np.array(list(product(sz_values, repeat=self.length)))
        return c[np.sum(c, axis=1) == m, :]

    # gives the list of configurations for length of n and in Sz=m.
    # it works faster than the 'configs' function for n > 9
    @lru_cache  # this memoize
    def rec_configs(self, m, n):
        if n == 2 and m == 0:
            return coo_matrix([[-1, 1], [0, 0], [1, -1]])
        elif n == 2 and m == 1:
            return coo_matrix([[0, 1], [1, 0]])
        elif n == 2 and m == -1:
            return coo_matrix([[-1, 0], [0, -1]])
        elif abs(m) == n:
            return coo_matrix([[np.sign(m)] * n])
        elif m == n - 1:
            c0 = hstack([coo_matrix([0] * hilbert_space_size(m, n - 1)).T, self.rec_configs(m, n - 1)])
            cm = hstack([coo_matrix([1] * hilbert_space_size(m - 1, n - 1)).T, self.rec_configs(m - 1, n - 1)])
            return bmat([[c0], [cm]])
        elif m == 1 - n:
            cp = hstack([coo_matrix([-1] * hilbert_space_size(m + 1, n - 1)).T, self.rec_configs(m + 1, n - 1)])
            c0 = hstack([coo_matrix([0] * hilbert_space_size(m, n - 1)).T, self.rec_configs(m, n - 1)])
            return bmat([[cp], [c0]])
        elif abs(m) < n - 1:
            cp = hstack([coo_matrix([-1] * hilbert_space_size(m + 1, n - 1)).T, self.rec_configs(m + 1, n - 1)])
            c0 = hstack([coo_matrix([0] * hilbert_space_size(m, n - 1)).T, self.rec_configs(m, n - 1)])
            cm = hstack([coo_matrix([1] * hilbert_space_size(m - 1, n - 1)).T, self.rec_configs(m - 1, n - 1)])
            return bmat([[cp], [c0], [cm]])
        else:
            return "invalid inputs"

    # the Hamiltonian for a single chain of length n, in the S^z=m subspace, and in the absence of magnetic fields
    @lru_cache  # this memoize
    def rec_chain_hamiltonian_szm(self, m, n):
        # n = self.length
        # chain = SingleChain(n - 1)  # the class for a chain of length n - 1
        if n == 2 and m == 0:
            return coo_matrix([
                [-1, 1, 0],
                [1, 0, 1],
                [0, 1, -1]
            ])
        elif n == 2 and abs(m) == 1:
            return coo_matrix([[0, 1], [1, 0]])
        elif abs(m) == n:
            return coo_matrix([[n - 1]])
        elif m == n - 1:
            h0 = self.rec_chain_hamiltonian_szm(m, n - 1)
            hm = self.rec_chain_hamiltonian_szm(m - 1, n - 1)
            dm = diags(np.concatenate(([-1] * hilbert_space_size(m, n - 2),
                                       [0] * hilbert_space_size(m - 1, n - 2),
                                       [1] * hilbert_space_size(m - 2, n - 2))))
            m0 = diags([1] * (hilbert_space_size(m - 1, n - 1) - hilbert_space_size(m - 2, n - 2)),
                       -hilbert_space_size(m + 1, n - 2),
                       shape=(hilbert_space_size(m, n - 1), hilbert_space_size(m - 1, n - 1)))
            return bmat([[h0, m0], [m0.T, hm + dm]])
        elif m == 1 - n:
            hp = self.rec_chain_hamiltonian_szm(m + 1, n - 1)
            dp = diags(np.concatenate(([1] * hilbert_space_size(m + 2, n - 2),
                                       [0] * hilbert_space_size(m + 1, n - 2),
                                       [-1] * hilbert_space_size(m, n - 2))))
            h0 = self.rec_chain_hamiltonian_szm(m, n - 1)
            p0 = diags([1] * (hilbert_space_size(m + 1, n - 1) - hilbert_space_size(m + 2, n - 2)),
                       -hilbert_space_size(m + 2, n - 2),
                       shape=(hilbert_space_size(m + 1, n - 1), hilbert_space_size(m, n - 1)))
            return bmat([[hp + dp, p0], [p0.T, h0]])
        elif abs(m) < n - 1:
            hp = self.rec_chain_hamiltonian_szm(m + 1, n - 1)
            dp = diags(np.concatenate(([1] * hilbert_space_size(m + 2, n - 2),
                                       [0] * hilbert_space_size(m + 1, n - 2),
                                       [-1] * hilbert_space_size(m, n - 2))))
            h0 = self.rec_chain_hamiltonian_szm(m, n - 1)
            hm = self.rec_chain_hamiltonian_szm(m - 1, n - 1)
            dm = diags(np.concatenate(([-1] * hilbert_space_size(m, n - 2),
                                       [0] * hilbert_space_size(m - 1, n - 2),
                                       [1] * hilbert_space_size(m - 2, n - 2))))
            p0 = diags([1] * (hilbert_space_size(m + 1, n - 1) - hilbert_space_size(m + 2, n - 2)),
                       -hilbert_space_size(m + 2, n - 2),
                       shape=(hilbert_space_size(m + 1, n - 1), hilbert_space_size(m, n - 1)))
            m0 = diags([1] * (hilbert_space_size(m - 1, n - 1) - hilbert_space_size(m - 2, n - 2)),
                       -hilbert_space_size(m + 1, n - 2),
                       shape=(hilbert_space_size(m, n - 1), hilbert_space_size(m - 1, n - 1)))
            return bmat([[hp + dp, p0, None],
                         [p0.T, h0, m0],
                         [None, m0.T, hm + dm]])
        else:
            return "invalid inputs"

    # the Hamiltonian for local magnetic field on a chain of length n and in the S^z=m subspace
    @lru_cache  # this memoize
    def local_field(self, m):
        n = self.length
        return diags(np.concatenate(([-1] * hilbert_space_size(m + 1, n - 1),
                                     [0] * hilbert_space_size(m, n - 1),
                                     [1] * hilbert_space_size(m - 1, n - 1))))

    # finds the exact unitary matrix exp(-iH*t) in the basis of S_tot^z = 0
    def unitary(self, b, t):
        n = self.length
        h = (self.rec_chain_hamiltonian_szm(0, n) + b * self.local_field(0)).tocsc()
        return linalg.expm(-1j * h * t).toarray()

    # finds the approximate unitary matrix exp(-iH*t) in the basis of first d states of H0
    @lru_cache  # this memoize
    def unitary_approx(self, b, t, d):
        h0 = self.rec_chain_hamiltonian_szm(0, self.length)
        h = (h0 + b * self.local_field(0)).tocsc()
        e, v = self.eigen_system(0, k=d)
        hf = v.T @ h @ v
        return linalg.expm(-1j * hf * t)

    # finds the k lowest eigenstates for length n and in S^z=m subspace, and the local field b
    @lru_cache  # this memoize
    def eigen_system(self, m=0, b=0, k=3, vectors=True):
        n = self.length
        h = self.rec_chain_hamiltonian_szm(m, n) + b * self.local_field(m)
        return linalg.eigsh(h, k, which='SA', return_eigenvectors=vectors)

    # finds the k lowest eigenstates for length n and in S^z=m subspace and their total S
    # in the absence of the local field
    @lru_cache  # this memoize
    def levels_s_tot(self, k=3):
        e, v = self.eigen_system(k=k)
        s_tot = self.rec_s_tot(0, self.length)
        s = np.round((v.T @ s_tot @ v).diagonal())
        return e, s

    # finds the k lowest eigenstates for length n, background field b0, and the local field b1
    # in the full Hilbert subspace
    @lru_cache  # this memoize
    def eigen_system_full(self, b0=0, b1=0, k=9, vectors=True):
        n = self.length
        h = self.rec_chain_hamiltonian_szm(-n, n)
        b = self.local_field(-n)
        bb = [-n]  # initiating the background magnetic field b0
        for m in range(-n + 1, n + 1):
            h = bmat([[h, None], [None, self.rec_chain_hamiltonian_szm(m, n)]])
            b = bmat([[b, None], [None, self.local_field(m)]])
            bb = hstack([bb, [m] * hilbert_space_size(m, n)])
        bb = diags(bb.toarray()[0, :])  # the background magnetic field
        # print(bb.shape)
        # print(3**n)
        return linalg.eigsh(h + b1 * b + b0 * bb, k, which='SA', return_eigenvectors=vectors)

    # finds the admixture to the states |0>' and |1>' for finite local field b
    def admix(self, b):
        e0, v0 = self.eigen_system(0)
        e, v = self.eigen_system(0, b)
        return 1 - sum((v[:, :2].T @ v0[:, :2]) ** 2)

    # finds <S^z> for a wavefunction v
    def sz_avg(self, m, v):
        return abs(v) ** 2 @ self.rec_configs(m, self.length)

    # the total angular momentum for a chain of length n and in the S^z=m subspace
    # built in a recursive way
    @lru_cache  # this memoize
    def rec_s_tot(self, m, n):
        if n == 2 and m == 0:
            return coo_matrix([
                [2, 2, 0],
                [2, 4, 2],
                [0, 2, 2]
            ])
        elif n == 2 and abs(m) == 1:
            return coo_matrix([[4, 2], [2, 4]])
        elif abs(m) == n:
            return coo_matrix([[n * (n + 1)]])
        elif abs(m) < n:
            if m == n - 1:
                s0 = self.rec_s_tot(m, n - 1) + diags([2] * hilbert_space_size(m, n - 1))
                sm = self.rec_s_tot(m - 1, n - 1) + diags([2 * m] * hilbert_space_size(m - 1, n - 1))
                xm = coo_matrix([[2] * (n - 1)])
                return bmat([[s0, xm], [xm.T, sm]])
            elif m == 1 - n:
                sp = self.rec_s_tot(m + 1, n - 1) - diags([2 * m] * hilbert_space_size(m + 1, n - 1))
                s0 = self.rec_s_tot(m, n - 1) + diags([2] * hilbert_space_size(m, n - 1))
                xp = coo_matrix([[2] * (n - 1)])
                return bmat([[sp, xp.T], [xp, s0]])
            elif m == n - 2:
                sp = self.rec_s_tot(m + 1, n - 1) - diags([2 * m] * hilbert_space_size(m + 1, n - 1))
                s0 = self.rec_s_tot(m, n - 1) + diags([2] * hilbert_space_size(m, n - 1))
                sm = self.rec_s_tot(m - 1, n - 1) + diags([2 * m] * hilbert_space_size(m - 1, n - 1))
                xmm = self.rec_s_tot(m - 1, n - 1).tolil()[
                      hilbert_space_size(m, n - 2):-hilbert_space_size(m - 2, n - 2),
                      -hilbert_space_size(m - 2, n - 2):]
                xp = coo_matrix([[2] * (n - 1)])
                dp = diags([2] * (n - 2), 1, shape=(n - 2, n - 1))
                xm = bmat([[xp, None], [dp, xmm]])
            elif m == 2 - n:
                sp = self.rec_s_tot(m + 1, n - 1) - diags([2 * m] * hilbert_space_size(m + 1, n - 1))
                s0 = self.rec_s_tot(m, n - 1) + diags([2] * hilbert_space_size(m, n - 1))
                sm = self.rec_s_tot(m - 1, n - 1) + diags([2 * m] * hilbert_space_size(m - 1, n - 1))
                xpp = self.rec_s_tot(m + 1, n - 1).tolil()[
                      :hilbert_space_size(m + 2, n - 2),
                      hilbert_space_size(m + 2, n - 2):-hilbert_space_size(m, n - 2)]
                xm = coo_matrix([[2] * (n - 1)]).T
                dm = diags([2] * (n - 2), shape=(n - 1, n - 2))
                xp = bmat([[xpp, None], [dm, xm]])
            elif m == 3 - n:
                sp = self.rec_s_tot(m + 1, n - 1) - diags([2 * m] * hilbert_space_size(m + 1, n - 1))
                s0 = self.rec_s_tot(m, n - 1) + diags([2] * hilbert_space_size(m, n - 1))
                sm = self.rec_s_tot(m - 1, n - 1) + diags([2 * m] * hilbert_space_size(m - 1, n - 1))
                xpp = self.rec_s_tot(m + 1, n - 1).tolil()[
                      :hilbert_space_size(m + 2, n - 2),
                      hilbert_space_size(m + 2, n - 2):-hilbert_space_size(m, n - 2)]
                xpm = self.rec_s_tot(m + 1, n - 1).tolil()[
                      hilbert_space_size(m + 2, n - 2):-hilbert_space_size(m, n - 2),
                      -hilbert_space_size(m, n - 2):]
                xmp = self.rec_s_tot(m - 1, n - 1).tolil()[
                      :hilbert_space_size(m, n - 2), hilbert_space_size(m, n - 2):]
                xmm = coo_matrix([[0]])
                dp = diags([2] * (hilbert_space_size(m + 1, n - 2) + hilbert_space_size(m, n - 2)),
                           -hilbert_space_size(m + 2, n - 2),
                           shape=(hilbert_space_size(m + 1, n - 1), hilbert_space_size(m, n - 1)))
                dm = diags([2] * (hilbert_space_size(m, n - 2) + hilbert_space_size(m - 1, n - 2)),
                           -hilbert_space_size(m + 1, n - 2),
                           shape=(hilbert_space_size(m, n - 1), hilbert_space_size(m - 1, n - 1)))
                xp = bmat([[xpp, None, None], [None, xpm, None], [None, None, xmp]]) + dp
                xm = bmat([[xpm, None], [None, xmp], [None, xmm]]) + dm
            elif m == n - 3:
                sp = self.rec_s_tot(m + 1, n - 1) - diags([2 * m] * hilbert_space_size(m + 1, n - 1))
                s0 = self.rec_s_tot(m, n - 1) + diags([2] * hilbert_space_size(m, n - 1))
                sm = self.rec_s_tot(m - 1, n - 1) + diags([2 * m] * hilbert_space_size(m - 1, n - 1))
                xpm = coo_matrix([[2] * (n - 1)])
                xmp = self.rec_s_tot(m - 1, n - 1).tolil()[
                      :hilbert_space_size(m, n - 2),
                      hilbert_space_size(m, n - 2):-hilbert_space_size(m - 2, n - 2)]
                xmm = self.rec_s_tot(m - 1, n - 1).tolil()[
                      hilbert_space_size(m, n - 2):-hilbert_space_size(m - 2, n - 2),
                      -hilbert_space_size(m - 2, n - 2):]
                dp = diags([2] * (n - 2), 1, shape=(n - 2, n - 1))
                dm = diags([2] * (hilbert_space_size(m, n - 2) + hilbert_space_size(m - 1, n - 2)),
                           -hilbert_space_size(m + 1, n - 2),
                           shape=(hilbert_space_size(m, n - 1), hilbert_space_size(m - 1, n - 1)))
                xp = bmat([[xpm, None], [dp, xmp]])
                xpm = coo_matrix([[2] * (n - 2)])
                xm = bmat([[xpm, None, None], [None, xmp, None], [None, None, xmm]]) + dm
            else:
                sp = self.rec_s_tot(m + 1, n - 1) - diags([2 * m] * hilbert_space_size(m + 1, n - 1))
                s0 = self.rec_s_tot(m, n - 1) + diags([2] * hilbert_space_size(m, n - 1))
                sm = self.rec_s_tot(m - 1, n - 1) + diags([2 * m] * hilbert_space_size(m - 1, n - 1))
                xpp = self.rec_s_tot(m + 1, n - 1).tolil()[
                      :hilbert_space_size(m + 2, n - 2),
                      hilbert_space_size(m + 2, n - 2):-hilbert_space_size(m, n - 2)]
                xpm = self.rec_s_tot(m + 1, n - 1).tolil()[
                      hilbert_space_size(m + 2, n - 2):-hilbert_space_size(m, n - 2),
                      -hilbert_space_size(m, n - 2):]
                xmp = self.rec_s_tot(m - 1, n - 1).tolil()[
                      :hilbert_space_size(m, n - 2),
                      hilbert_space_size(m, n - 2):-hilbert_space_size(m - 2, n - 2)]
                xmm = self.rec_s_tot(m - 1, n - 1).tolil()[
                      hilbert_space_size(m, n - 2):-hilbert_space_size(m - 2, n - 2),
                      -hilbert_space_size(m - 2, n - 2):]
                dp = diags([2] * (hilbert_space_size(m + 1, n - 2) + hilbert_space_size(m, n - 2)),
                           -hilbert_space_size(m + 2, n - 2),
                           shape=(hilbert_space_size(m + 1, n - 1), hilbert_space_size(m, n - 1)))
                dm = diags([2] * (hilbert_space_size(m, n - 2) + hilbert_space_size(m - 1, n - 2)),
                           -hilbert_space_size(m + 1, n - 2),
                           shape=(hilbert_space_size(m, n - 1), hilbert_space_size(m - 1, n - 1)))
                xp = bmat([[xpp, None, None], [None, xpm, None], [None, None, xmp]]) + dp
                xm = bmat([[xpm, None, None], [None, xmp, None], [None, None, xmm]]) + dm
            return bmat([[sp, xp, None],
                         [xp.T, s0, xm],
                         [None, xm.T, sm]])
        else:
            return "invalid inputs"

    # the total angular momentum for a chain of length n and in the full Hilbert space
    # built based on rec_s_tot
    @lru_cache  # this memoize
    def rec_s_tot_full(self):
        n = self.length
        s = self.rec_s_tot(-n, n)
        for m in range(-n + 1, n + 1):
            s = bmat([[s, None], [None, self.rec_s_tot(m, n)]])
        return s

    # computes S_tot^z of a vector v in the full Hilbert space
    def sz_tot_full(self, v):
        n = self.length
        s = np.concatenate(list(map(lambda m: [m] * hilbert_space_size(m, n), range(-n, n + 1))))
        return np.round(s @ v ** 2)


# takes energy 'e' and S_tot 's' of a pair and builds the sets of levels they generate
# the second column is ma + mb and the third ma - mb
def uncoupled_pair(e, s):
    l = (-1 + np.sqrt(1 + 4 * s)) / 2  # finds l from s = l(l+1) which is the output ot S_tot operator
    la = int(l[0])
    lb = int(l[1])
    levels = np.zeros(((2 * la + 1), (2 * lb + 1), 3))
    for ma in range(-la, la + 1):
        for mb in range(-lb, lb + 1):
            levels[ma, mb, :] = [e[0] + e[1], ma + mb, ma - mb]
    return np.reshape(levels, ((2 * la + 1) * (2 * lb + 1), 3))


# extracts the index of the computational basis from the output of the hamiltoninan_eff_sz and eigen_system_eff
def compute_basis(states):
    states = states[:, 0]
    mid = int(len(states) / 2)
    # print(states[mid])
    i00 = int((states[: mid] ** 2).sum())
    i01 = i00 + 1
    i10 = int(i00 + states[mid])
    i11 = i10 + 1
    return [i00, i01, i10, i11]


# class of all functions for a pair of chains
class TwoChains:

    def __init__(self, length):
        self.length = length

    # gives the full spectrum of a pair of uncoupled chains from the first k states of each chain
    # the second column is ma + mb
    # the third column is ma - mb
    @lru_cache  # this memoize
    def uncoupled_spect(self, k=3):
        chain = SingleChain(self.length)
        e, s = chain.levels_s_tot(k=k)
        ee = np.asarray(np.meshgrid(e, e)).T
        ss = np.asarray(np.meshgrid(s, s)).T
        levels = np.empty((0, 3))
        for i in range(k):
            for j in range(k):
                levels = np.vstack([levels, uncoupled_pair(ee[i, j], ss[i, j])])
        return levels

    # building the vector |VA VB> from VA and VB
    # such that the two first elements of the configuration correspond to the first two sites of the chain A and B
    # vector A is from Sz=m subspace and vector B is from Sz=n subspace
    def vector_mixer(self, m, n, va, vb):
        l = self.length
        # print(len(va) * len(vb))
        va = {-1: va[:hilbert_space_size(m + 1, l - 1)],
              0: va[hilbert_space_size(m + 1, l - 1): -hilbert_space_size(m - 1, l - 1)],
              1: va[-hilbert_space_size(m - 1, l - 1):]}
        vb = {-1: vb[:hilbert_space_size(n + 1, l - 1)],
              0: vb[hilbert_space_size(n + 1, l - 1): -hilbert_space_size(n - 1, l - 1)],
              1: vb[-hilbert_space_size(n - 1, l - 1):]}
        vab = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                vab = np.append(vab, np.kron(va[i], vb[j]))
        return vab

    # S-S+ operator on v. connecting (m+1, -m-1) pairs to (m,-m)
    # only used in the hamiltonian_slow version
    def smsp(self, m, v):
        n = self.length
        return np.concatenate(([0] * hilbert_space_size(m + 1, n - 1) * hilbert_space_size(-m + 1, n - 1),
                               v[hilbert_space_size(m + 2, n - 1) * hilbert_space_size(-m - 1, n):
                                 hilbert_space_size(m + 2, n - 1) * hilbert_space_size(-m - 1, n) +
                                 hilbert_space_size(m + 1, n - 1) * hilbert_space_size(-m, n - 1) +
                                 hilbert_space_size(m + 1, n - 1) * hilbert_space_size(-m - 1, n - 1)],
                               [0] * hilbert_space_size(m, n - 1) * hilbert_space_size(-m + 1, n - 1),
                               v[-hilbert_space_size(m, n - 1) * hilbert_space_size(-m - 1, n):
                                 -hilbert_space_size(m, n - 1) * hilbert_space_size(-m - 2, n - 1)],
                               [0] * hilbert_space_size(m - 1, n - 1) * hilbert_space_size(-m, n)))

    # this builds the full two chain hamiltonian in the configuration basis and in S_AB_tot^z=0 subspace
    # the output is 3 matrices for HA + HB, SA_tot - SB_tot, and SA1.SB1
    # it's not faster than the general one: hamiltonian_full_sz
    @lru_cache  # this memoize
    def hamiltonian_full(self):
        n = self.length
        chain = SingleChain(n)
        b_ab = [-n]  # initiate the diagonal BA - BB part
        # h0 initiates the uncoupled part H0A + H0B
        h0 = kronsum(chain.rec_chain_hamiltonian_szm(-n, n), chain.rec_chain_hamiltonian_szm(n, n))
        # initiates the coupling matrix
        sasb = coo_matrix(([-1, 1], ([0, 0], [0, n ** 2 - n + 1])), shape=(1, n ** 2 + 1))
        smsp = coo_matrix(([1], ([0], [n ** 2 - n])), shape=(1, n ** 2))
        for m in range(-n + 1, n + 1):
            t0 = time.time()
            # the BA - BB diagonal
            b_ab = np.append(b_ab, [m] * hilbert_space_size(m, n) ** 2)
            # print(b_ab.shape)
            # print(m)
            # the uncoupled HA + HB
            hp = chain.rec_chain_hamiltonian_szm(m, n)
            hm = chain.rec_chain_hamiltonian_szm(-m, n)
            h0 = bmat([[h0, None], [None, kronsum(hm, hp)]])
            # building the diagonal SAzSBz
            szsz = diags(np.concatenate(
                (list(-np.concatenate((
                    [-1] * hilbert_space_size(-m + 1, n - 1),
                    [0] * hilbert_space_size(-m, n - 1),
                    [1] * hilbert_space_size(-m - 1, n - 1)
                ))) * hilbert_space_size(m + 1, n - 1),
                 [0] * hilbert_space_size(m, n - 1) * hilbert_space_size(-m, n),
                 list(np.concatenate((
                     [-1] * hilbert_space_size(-m + 1, n - 1),
                     [0] * hilbert_space_size(-m, n - 1),
                     [1] * hilbert_space_size(-m - 1, n - 1)
                 ))) * hilbert_space_size(m - 1, n - 1)
                 )))
            # print(szsz.shape)
            t1 = time.time()
            print(m)
            print('diagonal', t1 - t0)
            # building the off-diagonal S-S+ and putting SA.SB together
            t0 = time.time()
            if m < n:
                smsp_m = smsp  # storing the previous S-S+
                # this shortcut method is based on the geometry of the matrix that I found analytically
                smsp = coo_matrix((hilbert_space_size(m - 1, n - 1),
                                   hilbert_space_size(m + 2, n - 1) * hilbert_space_size(m + 1, n)))
                for i in range(hilbert_space_size(m, n - 1) + hilbert_space_size(m + 1, n - 1)):
                    if i < hilbert_space_size(m, n - 1) + hilbert_space_size(m + 1, n - 1) - 1:
                        d = block_diag((
                            diags([1] * (hilbert_space_size(m, n - 1) + hilbert_space_size(m + 1, n - 1))),
                            coo_matrix((hilbert_space_size(m - 1, n - 1), hilbert_space_size(m + 2, n - 1)))
                        ))
                    else:
                        d = block_diag((
                            diags([1] * (hilbert_space_size(m, n - 1) + hilbert_space_size(m + 1, n - 1))),
                            coo_matrix((hilbert_space_size(m - 1, n - 1) * hilbert_space_size(m, n),
                                        hilbert_space_size(m + 2, n - 1)))
                        ))
                    smsp = block_diag((smsp, d))
                sasb = hstack([sasb, coo_matrix((sasb.shape[0], hilbert_space_size(m + 1, n) ** 2))])
                # this is the number of columns of the empty block needed on the left of the new row
                d = sasb.shape[1] - (
                    (hilbert_space_size(m, n) ** 2 +
                     hilbert_space_size(m - 1, n) ** 2 +
                     hilbert_space_size(m + 1, n) ** 2))
                sasb = vstack([sasb, hstack([coo_matrix((hilbert_space_size(m, n) ** 2, d)), smsp_m.T, szsz, smsp])])
                # print(sasb.nnz)
            else:
                # print(smsp.T.toarray())
                sasb = vstack([sasb,
                               coo_matrix(([1, -1], ([0, 0],
                                                     [hilbert_space_size(0, 2 * n) - n ** 2 + n - 2,
                                                      hilbert_space_size(0, 2 * n) - 1])),
                                          shape=(1, hilbert_space_size(0, 2 * n)))])
                # print(sasb.nnz)
            t1 = time.time()
            print('off-diagonal', t1 - t0)
        return h0, diags(b_ab), sasb

    # this builds the full two chain hamiltonian in the configuration basis and in S_AB_tot^z=m subspace
    # the output is 3 matrices for HA + HB, SA_tot - SB_tot, and SA1.SB1
    @lru_cache  # this memoize
    def hamiltonian_full_sz(self, m=0):
        n = self.length
        chain = SingleChain(n)
        if m > 0:
            ms = range(m - n, n + 1)
            print(np.asarray(ms))
            print(m - np.asarray(ms))
        else:
            ms = range(-n, n + m + 1)
            print(np.asarray(ms))
            print(m - np.asarray(ms))
        # initiate the diagonal BA - BB part
        b_ab = [ms[0] - m / 2] * hilbert_space_size(ms[0], n) * hilbert_space_size(m - ms[0], n)
        # initiates the uncoupled part H0A + H0B
        ha = chain.rec_chain_hamiltonian_szm(ms[0], n)
        hb = chain.rec_chain_hamiltonian_szm(m - ms[0], n)
        h0 = kronsum(hb, ha)
        # initial S1Az * S1Bz
        szsz = diags(np.concatenate(
            (list(-np.concatenate((
                [-1] * hilbert_space_size(m - ms[0] + 1, n - 1),
                [0] * hilbert_space_size(m - ms[0], n - 1),
                [1] * hilbert_space_size(m - ms[0] - 1, n - 1)
            ))) * hilbert_space_size(ms[0] + 1, n - 1),
             [0] * hilbert_space_size(ms[0], n - 1) * hilbert_space_size(m - ms[0], n),
             list(np.concatenate((
                 [-1] * hilbert_space_size(m - ms[0] + 1, n - 1),
                 [0] * hilbert_space_size(m - ms[0], n - 1),
                 [1] * hilbert_space_size(m - ms[0] - 1, n - 1)
             ))) * hilbert_space_size(ms[0] - 1, n - 1)
             )))
        # initial S-Az * S+Bz
        smsp = coo_matrix((hilbert_space_size(m - ms[0] + 1, n - 1),
                           hilbert_space_size(ms[0] + 2, n - 1) * hilbert_space_size(m - ms[0] - 1, n)))
        for i in range(hilbert_space_size(ms[0], n - 1) + hilbert_space_size(ms[0] + 1, n - 1)):
            # print('#', i)
            if i < hilbert_space_size(ms[0], n - 1) + hilbert_space_size(ms[0] + 1, n - 1) - 1:
                d = block_diag((
                    diags([1] * (hilbert_space_size(m - ms[0], n - 1) + hilbert_space_size(m - ms[0] - 1, n - 1))),
                    coo_matrix((hilbert_space_size(m - ms[0] + 1, n - 1), hilbert_space_size(m - ms[0] - 2, n - 1)))
                ))
            else:
                d = block_diag((
                    diags([1] * (hilbert_space_size(m - ms[0], n - 1) + hilbert_space_size(m - ms[0] - 1, n - 1))),
                    coo_matrix((hilbert_space_size(ms[0] - 1, n - 1) * hilbert_space_size(m - ms[0], n),
                                hilbert_space_size(m - ms[0] - 2, n - 1)))
                ))
                # print(d.shape)
            smsp = block_diag((smsp, d))
        sasb = hstack([szsz, smsp])
        for ma in ms[1:]:
            mb = m - ma
            # the BA - BB diagonal
            b_ab = np.append(b_ab, [ma - m / 2] * hilbert_space_size(ma, n) * hilbert_space_size(mb, n))
            # the uncoupled HA + HB
            ha = chain.rec_chain_hamiltonian_szm(ma, n)
            hb = chain.rec_chain_hamiltonian_szm(mb, n)
            h0 = bmat([[h0, None], [None, kronsum(hb, ha)]])
            # print(kronsum(hb, ha).shape)
            # print(ma, h0.shape, b_ab.shape)
            # building the diagonal S1Az*S1Bz
            szsz = diags(np.concatenate(
                (list(-np.concatenate((
                    [-1] * hilbert_space_size(mb + 1, n - 1),
                    [0] * hilbert_space_size(mb, n - 1),
                    [1] * hilbert_space_size(mb - 1, n - 1)
                ))) * hilbert_space_size(ma + 1, n - 1),
                 [0] * hilbert_space_size(ma, n - 1) * hilbert_space_size(mb, n),
                 list(np.concatenate((
                     [-1] * hilbert_space_size(mb + 1, n - 1),
                     [0] * hilbert_space_size(mb, n - 1),
                     [1] * hilbert_space_size(mb - 1, n - 1)
                 ))) * hilbert_space_size(ma - 1, n - 1)
                 )))
            # building the off-diagonal S-S+ and putting SA.SB together
            if ma < ms[-1]:
                smsp_m = smsp  # storing the previous S-S+
                # this shortcut method is based on the geometry of the matrix that I found analytically
                smsp = coo_matrix((hilbert_space_size(mb + 1, n - 1),
                                   hilbert_space_size(ma + 2, n - 1) * hilbert_space_size(mb - 1, n)))
                for i in range(hilbert_space_size(ma, n - 1) + hilbert_space_size(ma + 1, n - 1)):
                    if i < hilbert_space_size(ma, n - 1) + hilbert_space_size(ma + 1, n - 1) - 1:
                        d = block_diag((
                            diags([1] * (hilbert_space_size(mb, n - 1) + hilbert_space_size(mb - 1, n - 1))),
                            coo_matrix((hilbert_space_size(mb + 1, n - 1), hilbert_space_size(mb - 2, n - 1)))
                        ))
                    else:
                        d = block_diag((
                            diags([1] * (hilbert_space_size(mb, n - 1) + hilbert_space_size(mb - 1, n - 1))),
                            coo_matrix((hilbert_space_size(ma - 1, n - 1) * hilbert_space_size(mb, n),
                                        hilbert_space_size(mb - 2, n - 1)))
                        ))
                    smsp = block_diag((smsp, d))
                # print(szsz.shape, smsp.shape)
                sasb = hstack([sasb, coo_matrix((sasb.shape[0],
                                                 hilbert_space_size(ma + 1, n) * hilbert_space_size(mb - 1, n)))])
                # this is the number of columns of the empty block needed on the left of the new row
                d = sasb.shape[1] - (
                        hilbert_space_size(ma, n) * hilbert_space_size(mb, n) +
                        hilbert_space_size(ma + 1, n) * hilbert_space_size(mb - 1, n) +
                        hilbert_space_size(ma - 1, n) * hilbert_space_size(mb + 1, n))
                sasb = vstack([sasb,
                               hstack([coo_matrix((hilbert_space_size(ma, n) * hilbert_space_size(mb, n), d)),
                                       smsp_m.T, szsz, smsp])])
                # print(ma, sasb.shape)
            else:
                smsp_m = smsp  # storing the previous S-S+
                sasb = vstack([sasb,
                               hstack(
                                   [coo_matrix(
                                       (szsz.shape[0], hilbert_space_size(m, 2 * n) - szsz.shape[0] - smsp.shape[0])),
                                       smsp_m.T, szsz])])
                # print(ma, sasb.shape)
        # print(h0.shape, diags(b_ab).shape)
        return h0, diags(b_ab), sasb

    # THIS ESSENTIALLY HAS NO ADVANTAGE OVER THE GENERAL ONE: hamiltonian_eff_sz
    # generates the effective two chain Hamiltonian in the basis of pair states of two single chain |A B >.
    # S_tot^z = SA_tot^z + SB_tot^z = 0, the subspace of the computational basis.
    # k is the number of single chain states in S_tot^z = 0 subspace.
    # States from other subspaces are chosen such that
    # they are in the same energy bound determined by S_tot^z = 0 subspace.
    # the output is the diagonal of the uncoupled bases, the diagonal terms, BA - BB, the S1A.S1B matrix element matrix,
    # and the number of single chain states used.
    @lru_cache  # this memoize
    def hamiltonian_eff(self, k=3):
        n = self.length
        chain = SingleChain(n)
        # starts with Sz = 0 subspace
        t0 = time.time()
        e, v = chain.eigen_system(k=k)
        s_tot = chain.rec_s_tot(0, n)  # the S_tot operator in m=0 subspace
        s = np.round((v.T @ s_tot @ v).diagonal())  # S_tot eigenvalues = l(l + 1)
        l = (-1 + np.sqrt(1 + 4 * s)) / 2  # finds l from s = l(l+1) which is the output ot S_tot operator
        t1 = time.time()
        print('single chain states and S_tot in Sz=0', t1 - t0)

        # the diagonal made out of EA + EB. both A and B from Sz = 0
        all_e = {0: np.add.outer(e, e).reshape(k ** 2)}

        # finds how many of each S_tot
        count = collections.Counter(l)
        # determines how many states should be considered in the subspace Sz = m
        mk = np.cumsum(list(count.values()))
        # print(mk)
        b_ab_len = np.append(k, k - mk)  # the size of (m, -m) block
        # print(b_ab_len)
        m_max = len(mk) - 1

        # building the BA - BB diagonal
        b_ab = []
        for i in range(-m_max, m_max + 1):
            b_ab = np.append(b_ab, [i] * (b_ab_len[abs(i)]) ** 2)
            # print(b_ab.shape)

        # builds |VA VB> vectors out of each pair of v
        # t0 = time.time()
        # print('test kron', np.kron(v, v).shape)
        # print(v.shape)
        # t1 = time.time()
        # print('kron of the bases takes', t1 - t0)
        t0 = time.time()
        all_v = {0: np.zeros((len(v) ** 2, k, k))}
        for i in range(k):
            for j in range(k):
                all_v[0][:, i, j] = self.vector_mixer(0, 0, v[:, i], v[:, j])
        all_v[0] = np.reshape(all_v[0], (len(v) ** 2, k ** 2))
        t1 = time.time()
        print('mixing states in Sz=0', t1 - t0)

        # building the diagonal SzSz matrix for m = 0
        t0 = time.time()
        szsz = diags(np.concatenate(([1] * hilbert_space_size(1, n - 1) ** 2,
                                     [0] * hilbert_space_size(1, n - 1) * hilbert_space_size(0, n - 1),
                                     [-1] * hilbert_space_size(1, n - 1) ** 2,
                                     [0] * hilbert_space_size(0, n - 1) * hilbert_space_size(0, n),
                                     [-1] * hilbert_space_size(1, n - 1) ** 2,
                                     [0] * hilbert_space_size(1, n - 1) * hilbert_space_size(0, n - 1),
                                     [1] * hilbert_space_size(1, n - 1) ** 2)))
        t1 = time.time()
        print('building the diagonal SzSz matrix for Sz = 0', t1 - t0)

        # builds S1A.S1B matrix elements between (m, -m) = (0, 0) pairs
        t0 = time.time()
        all_szsz = {0: all_v[0].T @ szsz @ all_v[0]}
        t1 = time.time()
        print('building effective SzSz in SzSz=(0,0)', t1 - t0)

        # finds the eigen vectors for subspaces with Sz = +m and -m and builds the SzSz diagonal blocks
        for m in range(1, len(mk)):
            # print(k - mk[m - 1])
            t0 = time.time()
            ep, vp = chain.eigen_system(m=m, k=k - mk[m - 1])
            em, vm = chain.eigen_system(m=-m, k=k - mk[m - 1])
            # builds the diagonal EA + EB for SzA = m and SzB = -m and reverse
            all_e[m] = np.add.outer(em, ep).reshape((k - mk[m - 1]) ** 2)
            all_e[-m] = np.add.outer(ep, em).reshape((k - mk[m - 1]) ** 2)
            # builds |VA VB> vectors out of each pair of VA from Sz = m and VB from Sz = -m and reverse
            all_v[m] = np.zeros((len(vp) * len(vm), k - mk[m - 1], k - mk[m - 1]))
            all_v[-m] = np.zeros((len(vm) * len(vp), k - mk[m - 1], k - mk[m - 1]))
            for i in range(k - mk[m - 1]):
                for j in range(k - mk[m - 1]):
                    all_v[m][:, i, j] = self.vector_mixer(m, -m, vp[:, i], vm[:, j])
                    all_v[-m][:, i, j] = self.vector_mixer(-m, m, vm[:, i], vp[:, j])
            all_v[m] = np.reshape(all_v[m], (len(vp) * len(vm), (k - mk[m - 1]) ** 2))
            all_v[-m] = np.reshape(all_v[-m], (len(vm) * len(vp), (k - mk[m - 1]) ** 2))
            t1 = time.time()
            print('|m| =', m, 'vectors built. took', t1 - t0)
            # building the diagonal SzSz matrix for (m, -m) pairs
            t0 = time.time()
            szszp = diags(np.concatenate(([1] * hilbert_space_size(m + 1, n - 1) * hilbert_space_size(-m + 1, n - 1),
                                          [0] * hilbert_space_size(m + 1, n - 1) * hilbert_space_size(-m, n - 1),
                                          [-1] * hilbert_space_size(m + 1, n - 1) * hilbert_space_size(-m - 1, n - 1),
                                          [0] * hilbert_space_size(m, n - 1) * hilbert_space_size(-m, n),
                                          [-1] * hilbert_space_size(m - 1, n - 1) * hilbert_space_size(-m + 1, n - 1),
                                          [0] * hilbert_space_size(m - 1, n - 1) * hilbert_space_size(-m, n - 1),
                                          [1] * hilbert_space_size(m - 1, n - 1) * hilbert_space_size(-m - 1, n - 1))))
            szszm = diags(np.concatenate(([1] * hilbert_space_size(-m + 1, n - 1) * hilbert_space_size(m + 1, n - 1),
                                          [0] * hilbert_space_size(-m + 1, n - 1) * hilbert_space_size(m, n - 1),
                                          [-1] * hilbert_space_size(-m + 1, n - 1) * hilbert_space_size(m - 1, n - 1),
                                          [0] * hilbert_space_size(-m, n - 1) * hilbert_space_size(m, n),
                                          [-1] * hilbert_space_size(-m - 1, n - 1) * hilbert_space_size(m + 1, n - 1),
                                          [0] * hilbert_space_size(-m - 1, n - 1) * hilbert_space_size(m, n - 1),
                                          [1] * hilbert_space_size(-m - 1, n - 1) * hilbert_space_size(m - 1, n - 1))))
            # builds S1A.S1B matrix elements between (m, -m) pairs
            all_szsz[m] = all_v[m].T @ szszp @ all_v[m]
            all_szsz[-m] = all_v[-m].T @ szszm @ all_v[-m]
            t1 = time.time()
            print('building effective SzSz in SzSz=', (m, -m), 'takes', t1 - t0)

        # orders the all_e dic from -m to m
        all_e = dict(sorted(all_e.items()))
        # builds the full diagonal EA + EB by merging all_E
        diagonal_e = np.concatenate(list(all_e.values()))

        # building S-S+ off-diagonal matrices and put the final compact S1A.S1B matrix together
        all_smsp = {}
        dim = 2 * sum(b_ab_len ** 2) - k ** 2
        sasb = np.zeros((dim, dim))
        row = 0
        col = b_ab_len[m_max] ** 2 + b_ab_len[m_max - 1] ** 2
        for m in range(-m_max, m_max):
            # building the sparse off-diagonal matrix connecting m and m + 1
            t0 = time.time()
            u = diags([1] * (hilbert_space_size(m + 1, n - 1) * hilbert_space_size(-m - 1, n - 1) +
                             hilbert_space_size(m + 1, n - 1) * hilbert_space_size(-m, n - 1)))
            d = diags([1] * (hilbert_space_size(m, n - 1) * hilbert_space_size(-m - 1, n - 1) +
                             hilbert_space_size(m, n - 1) * hilbert_space_size(-m, n - 1)))
            z0 = coo_matrix((hilbert_space_size(m, n - 1) * hilbert_space_size(-m + 1, n - 1),
                             hilbert_space_size(m + 1, n - 1) * hilbert_space_size(-m - 2, n - 1)))
            zp = coo_matrix((hilbert_space_size(m + 1, n - 1) * hilbert_space_size(-m + 1, n - 1),
                             hilbert_space_size(m + 2, n - 1) * hilbert_space_size(-m - 1, n)))
            zm = coo_matrix((hilbert_space_size(m - 1, n - 1) * hilbert_space_size(-m, n),
                             hilbert_space_size(m, n - 1) * hilbert_space_size(-m - 2, n - 1)))
            mp = bmat([[u, None, None], [None, z0, None], [None, None, d]])
            mp = bmat([[zp, None, None], [None, mp, None], [None, None, zm]])
            # print(mp.shape)
            # print(all_v[m].shape)
            # print(all_v[m + 1].shape)
            # print(m)
            # forming S-S+
            all_smsp[m] = all_v[m].T @ mp @ all_v[m + 1]
            t1 = time.time()
            print('building effective S-S+ between', (m, m + 1), 'takes', t1 - t0)

            # putting the final compact S1A.S1B matrix together
            if m == -m_max:
                sasb[: b_ab_len[abs(m)] ** 2, : col] = np.concatenate((all_szsz[m], all_smsp[m]), axis=1)
            else:
                sasb[row: row + b_ab_len[abs(m)] ** 2, row - b_ab_len[abs(m - 1)] ** 2: col] = \
                    np.concatenate((all_smsp[m - 1].T, all_szsz[m], all_smsp[m]), axis=1)
            row = row + b_ab_len[abs(m)] ** 2
            col = col + b_ab_len[abs(m + 2)] ** 2
            # print('row', row)
            # print('column', col)
        sasb[row:, row - b_ab_len[abs(m_max - 1)] ** 2:] = \
            np.concatenate((all_smsp[m_max - 1].T, all_szsz[m_max]), axis=1)  # adds the last row

        return diagonal_e, b_ab, sasb, 2 * sum(b_ab_len) - k

    # this generates the effective hamiltonian in the subspace S_tot^z = S_A_tot^z + S_A_tot^z = m.
    # k is the number of states considered from S_A(B)_tot^z = 0 subspace
    # based on which single chain states from other subspaces are picked
    # it gives 4 outputs: Diagonals EA + EB, BA - BB, S1A.S1B matrix,
    # and the number of single chain states used (which is larger than k)
    @lru_cache  # this memoize
    def hamiltonian_eff_sz(self, m=0, k=3):
        n = self.length
        chain = SingleChain(n)
        # starts with Sz = 0 subspace
        t0 = time.time()
        e, v = chain.eigen_system(k=k)
        s_tot = chain.rec_s_tot(0, n)  # the S_tot operator in m=0 subspace
        s = np.round((v.T @ s_tot @ v).diagonal())  # S_tot eigenvalues = l(l + 1)
        l = (-1 + np.sqrt(1 + 4 * s)) / 2  # finds l from s = l(l+1) which is the output ot S_tot operator
        t1 = time.time()
        # print('single chain states and S_tot in Sz=0', t1 - t0)
        # finds how many of each S_tot
        count = collections.Counter(l)
        # determining how many states should be considered in the subspace Sz = m
        mk = np.cumsum(list(count.values()))
        # print(l)
        # print(mk)
        # the number of the considered single chain states with S_tot^z = m
        m_len = np.append(k, k - mk)
        # print(m_len)
        m_max = len(mk) - 1
        # print(m_max)
        if m > 0:
            ms = range(max(-m_max, m - m_max), m_max + 1)
            print(np.asarray(ms))
            # print(m - np.asarray(ms))
        else:
            ms = range(-m_max, min(m_max, m + m_max) + 1)
            print(np.asarray(ms))
            # print(m - np.asarray(ms))
        diagonal_e = []
        b_ab = []
        all_v = {}
        all_szsz = {}
        states = []
        for ma in ms:
            # counting the number of single chains states used
            states = np.append(states, [m_len[abs(ma)], m_len[abs(m-ma)]])
            # print(m_len[abs(ma)], m_len[abs(m - ma)], states, 2 * sum(m_len) - k)
            # building the BA - BB diagonal
            b_ab = np.append(b_ab, [ma - m / 2] * (m_len[abs(ma)] * m_len[abs(m - ma)]))
            # finding the single chain states pairs
            ea, va = chain.eigen_system(m=ma, k=m_len[abs(ma)])
            eb, vb = chain.eigen_system(m=m - ma, k=m_len[abs(m - ma)])
            # building the EA + EB diagonal
            diagonal_e = np.append(diagonal_e, np.add.outer(ea, eb).reshape((len(ea) * len(eb))))
            # mixing VA and VB to build |VA VB>
            all_v[ma] = np.zeros((len(va) * len(vb), m_len[abs(ma)], m_len[abs(m - ma)]))
            for i in range(m_len[abs(ma)]):
                for j in range(m_len[abs(m - ma)]):
                    all_v[ma][:, i, j] = self.vector_mixer(ma, m - ma, va[:, i], vb[:, j])
            all_v[ma] = np.reshape(all_v[ma], (len(va) * len(vb), m_len[abs(ma)] * m_len[abs(m - ma)]))
            # print(ma, all_v[ma].shape)
            # building the SzSz diagonal operator
            szsz = diags(
                np.concatenate(([1] * hilbert_space_size(ma + 1, n - 1) * hilbert_space_size(m - ma + 1, n - 1),
                                [0] * hilbert_space_size(ma + 1, n - 1) * hilbert_space_size(m - ma, n - 1),
                                [-1] * hilbert_space_size(ma + 1, n - 1) * hilbert_space_size(m - ma - 1, n - 1),
                                [0] * hilbert_space_size(ma, n - 1) * hilbert_space_size(m - ma, n),
                                [-1] * hilbert_space_size(ma - 1, n - 1) * hilbert_space_size(m - ma + 1, n - 1),
                                [0] * hilbert_space_size(ma - 1, n - 1) * hilbert_space_size(m - ma, n - 1),
                                [1] * hilbert_space_size(ma - 1, n - 1) * hilbert_space_size(m - ma - 1, n - 1))))
            # print(szsz.shape)
            # build the S1zA S1zB coupling matrix (the diagonal block) in |VA VB> basis
            all_szsz[ma] = all_v[ma].T @ szsz @ all_v[ma]
            # print(all_szsz[ma].shape)
        all_smsp = {}
        # initial setup for the final coupling matrices
        sasb = np.zeros((len(b_ab), len(b_ab)))
        row = 0
        col = m_len[abs(ms[0])] * m_len[abs(m - ms[0])] + m_len[abs(ms[1])] * m_len[abs(m - ms[1])]
        for ma in ms[:-1]:
            # building the S-S+ 0ff-diagonal operator
            u = diags([1] * (hilbert_space_size(ma + 1, n - 1) * hilbert_space_size(m - ma - 1, n - 1) +
                             hilbert_space_size(ma + 1, n - 1) * hilbert_space_size(m - ma, n - 1)))
            d = diags([1] * (hilbert_space_size(ma, n - 1) * hilbert_space_size(m - ma - 1, n - 1) +
                             hilbert_space_size(ma, n - 1) * hilbert_space_size(m - ma, n - 1)))
            z0 = coo_matrix((hilbert_space_size(ma, n - 1) * hilbert_space_size(m - ma + 1, n - 1),
                             hilbert_space_size(ma + 1, n - 1) * hilbert_space_size(m - ma - 2, n - 1)))
            zp = coo_matrix((hilbert_space_size(ma + 1, n - 1) * hilbert_space_size(m - ma + 1, n - 1),
                             hilbert_space_size(ma + 2, n - 1) * hilbert_space_size(m - ma - 1, n)))
            zm = coo_matrix((hilbert_space_size(ma - 1, n - 1) * hilbert_space_size(m - ma, n),
                             hilbert_space_size(ma, n - 1) * hilbert_space_size(m - ma - 2, n - 1)))
            smsp = bmat([[u, None, None], [None, z0, None], [None, None, d]])
            smsp = bmat([[zp, None, None], [None, smsp, None], [None, None, zm]])
            # build the S1-A S1+B coupling matrix (the off-diagonal block) in |VA VB> basis
            # print(ma, smsp.shape)
            # print(all_v[ma].shape, all_v[ma + 1].shape)
            all_smsp[ma] = all_v[ma].T @ smsp @ all_v[ma + 1]
            # print(ma, all_smsp[ma].shape)
            # putting the final compact S1A.S1B matrix together
            if ma == ms[0]:
                # the first row
                sasb[: m_len[abs(ma)] * m_len[abs(m - ma)], : col] = \
                    np.concatenate((all_szsz[ma], all_smsp[ma]), axis=1)
            else:
                sasb[row: row + m_len[abs(ma)] * m_len[abs(m - ma)],
                     row - m_len[abs(ma - 1)] * m_len[abs(m - ma + 1)]: col] = \
                    np.concatenate((all_smsp[ma - 1].T, all_szsz[ma], all_smsp[ma]), axis=1)
            row = row + m_len[abs(ma)] * m_len[abs(m - ma)]
            col = col + m_len[abs(ma + 2)] * m_len[abs(m - ma - 2)]
            # print('row', row)
            # print('column', col)
        # adding the last row
        sasb[row:, row - m_len[abs(ms[-2])] * m_len[abs(m - ms[-2])]:] = \
            np.concatenate((all_smsp[ms[-2]].T, all_szsz[ms[-1]]), axis=1)
        return diagonal_e, b_ab, sasb, states.reshape(len(ms), 2)

    # this is the same as hamiltonian_eff but without vector mixing
    # it is slower in making the S-S+ blocks
    @lru_cache  # this memoize
    def hamiltonian_eff_2(self, k=3):
        n = self.length
        chain = SingleChain(n)
        # starts with Sz = 0 subspace
        t0 = time.time()
        e, v = chain.eigen_system(k=k)
        s_tot = chain.rec_s_tot(0, n)  # the S_tot operator in m=0 subspace
        s = np.round((v.T @ s_tot @ v).diagonal())  # S_tot eigenvalues = l(l + 1)
        l = (-1 + np.sqrt(1 + 4 * s)) / 2  # finds l from s = l(l+1) which is the output ot S_tot operator
        t1 = time.time()
        print('single chain states and S_tot in Sz=0', t1 - t0)

        # the diagonal made out of EA + EB. both A and B from Sz = 0
        all_e = {0: np.add.outer(e, e).reshape(k ** 2)}

        # finds how many of each S_tot
        count = collections.Counter(l)
        # determines how many states should be considered in the subspace Sz = m
        mk = np.cumsum(list(count.values()))
        # print(mk)
        b_ab_len = np.append(k, k - mk)  # the size of (m, -m) block
        # print(b_ab_len)
        m_max = len(mk) - 1

        # building the BA - BB diagonal
        b_ab = []
        for i in range(-m_max, m_max + 1):
            b_ab = np.append(b_ab, [i] * (b_ab_len[abs(i)]) ** 2)
            # print(b_ab.shape)

        # builds |VA VB> vectors out of each pair of v
        t0 = time.time()
        all_v = {0: np.kron(v, v)}
        t1 = time.time()
        print('mixing states in Sz=0', t1 - t0)

        # building the diagonal SzSz matrix for m = 0
        t0 = time.time()
        szsz = diags(np.concatenate(
            (list(-np.concatenate((
                [-1] * hilbert_space_size(1, n - 1),
                [0] * hilbert_space_size(0, n - 1),
                [1] * hilbert_space_size(1, n - 1)
            ))) * hilbert_space_size(1, n - 1),
             [0] * hilbert_space_size(0, n - 1) * hilbert_space_size(0, n),
             list(np.concatenate((
                 [-1] * hilbert_space_size(1, n - 1),
                 [0] * hilbert_space_size(0, n - 1),
                 [1] * hilbert_space_size(1, n - 1)
             ))) * hilbert_space_size(1, n - 1)
             )))
        t1 = time.time()
        print('building the diagonal SzSz matrix for Sz = 0', t1 - t0)

        # builds S1A.S1B matrix elements between (m, -m) = (0, 0) pairs
        t0 = time.time()
        all_szsz = {0: all_v[0].T @ szsz @ all_v[0]}
        t1 = time.time()
        print('building effective SzSz in SzSz=(0,0)', t1 - t0)

        # finds the eigen vectors for subspaces with Sz = +m and -m and builds the SzSz diagonal blocks
        for m in range(1, len(mk)):
            # print(k - mk[m - 1])
            t0 = time.time()
            ep, vp = chain.eigen_system(m=m, k=k - mk[m - 1])
            em, vm = chain.eigen_system(m=-m, k=k - mk[m - 1])
            # builds the diagonal EA + EB for SzA = m and SzB = -m and reverse
            all_e[m] = np.add.outer(em, ep).reshape((k - mk[m - 1]) ** 2)
            all_e[-m] = np.add.outer(ep, em).reshape((k - mk[m - 1]) ** 2)
            # builds |VA VB> vectors out of each pair of VA from Sz = m and VB from Sz = -m and reverse
            all_v[m] = np.kron(vp, vm)
            all_v[-m] = np.kron(vm, vp)
            t1 = time.time()
            print('|m| =', m, 'vectors built. took', t1 - t0)
            # building the diagonal SzSz matrix for (m, -m) pairs
            t0 = time.time()
            szszp = diags(np.concatenate(
                (list(-np.concatenate((
                    [-1] * hilbert_space_size(-m + 1, n - 1),
                    [0] * hilbert_space_size(-m, n - 1),
                    [1] * hilbert_space_size(-m - 1, n - 1)
                ))) * hilbert_space_size(m + 1, n - 1),
                 [0] * hilbert_space_size(m, n - 1) * hilbert_space_size(-m, n),
                 list(np.concatenate((
                     [-1] * hilbert_space_size(-m + 1, n - 1),
                     [0] * hilbert_space_size(-m, n - 1),
                     [1] * hilbert_space_size(-m - 1, n - 1)
                 ))) * hilbert_space_size(m - 1, n - 1)
                 )))
            szszm = diags(np.concatenate(
                (list(-np.concatenate((
                    [-1] * hilbert_space_size(m + 1, n - 1),
                    [0] * hilbert_space_size(m, n - 1),
                    [1] * hilbert_space_size(m - 1, n - 1)
                ))) * hilbert_space_size(-m + 1, n - 1),
                 [0] * hilbert_space_size(-m, n - 1) * hilbert_space_size(m, n),
                 list(np.concatenate((
                     [-1] * hilbert_space_size(m + 1, n - 1),
                     [0] * hilbert_space_size(m, n - 1),
                     [1] * hilbert_space_size(m - 1, n - 1)
                 ))) * hilbert_space_size(-m - 1, n - 1)
                 )))
            # builds S1A.S1B matrix elements between (m, -m) pairs
            all_szsz[m] = all_v[m].T @ szszp @ all_v[m]
            all_szsz[-m] = all_v[-m].T @ szszm @ all_v[-m]
            t1 = time.time()
            print('building effective SzSz in SzSz=', (m, -m), 'takes', t1 - t0)

        # orders the all_e dic from -m to m
        all_e = dict(sorted(all_e.items()))
        # builds the full diagonal EA + EB by merging all_E
        diagonal_e = np.concatenate(list(all_e.values()))

        # building S-S+ off-diagonal matrices and put the final compact S1A.S1B matrix together
        all_smsp = {}
        dim = 2 * sum(b_ab_len ** 2) - k ** 2
        sasb = np.zeros((dim, dim))
        row = 0
        col = b_ab_len[m_max] ** 2 + b_ab_len[m_max - 1] ** 2
        for m in range(-m_max, m_max):
            # building the sparse off-diagonal matrix connecting m and m + 1
            t0 = time.time()
            smsp = coo_matrix((hilbert_space_size(m - 1, n - 1),
                               hilbert_space_size(m + 2, n - 1) * hilbert_space_size(m + 1, n)))
            for i in range(hilbert_space_size(m, n - 1) + hilbert_space_size(m + 1, n - 1)):
                if i < hilbert_space_size(m, n - 1) + hilbert_space_size(m + 1, n - 1) - 1:
                    d = block_diag((
                        diags([1] * (hilbert_space_size(m, n - 1) + hilbert_space_size(m + 1, n - 1))),
                        coo_matrix((hilbert_space_size(m - 1, n - 1), hilbert_space_size(m + 2, n - 1)))
                    ))
                else:
                    d = block_diag((
                        diags([1] * (hilbert_space_size(m, n - 1) + hilbert_space_size(m + 1, n - 1))),
                        coo_matrix((hilbert_space_size(m - 1, n - 1) * hilbert_space_size(m, n),
                                    hilbert_space_size(m + 2, n - 1)))
                    ))
                smsp = block_diag((smsp, d))
            # print(smsp.shape)
            # print(all_v[m].shape)
            # print(all_v[m + 1].shape)
            # print(m)
            # forming S-S+
            all_smsp[m] = all_v[m].T @ smsp @ all_v[m + 1]
            t1 = time.time()
            print('building effective S-S+ between', (m, m + 1), 'takes', t1 - t0)

            # putting the final compact S1A.S1B matrix together
            if m == -m_max:
                sasb[: b_ab_len[abs(m)] ** 2, : col] = np.concatenate((all_szsz[m], all_smsp[m]), axis=1)
            else:
                sasb[row: row + b_ab_len[abs(m)] ** 2, row - b_ab_len[abs(m - 1)] ** 2: col] = \
                    np.concatenate((all_smsp[m - 1].T, all_szsz[m], all_smsp[m]), axis=1)
            row = row + b_ab_len[abs(m)] ** 2
            col = col + b_ab_len[abs(m + 2)] ** 2
            # print('row', row)
            # print('column', col)
        sasb[row:, row - b_ab_len[abs(m_max - 1)] ** 2:] = \
            np.concatenate((all_smsp[m_max - 1].T, all_szsz[m_max]), axis=1)  # adds the last row

        return diagonal_e, b_ab, sasb, 2 * sum(b_ab_len) - k

    # this is the slow (earlier) version of hamiltonian_eff. I keep it in case because it might use less memory.
    @lru_cache  # this memoize
    def hamiltonian_slow(self, k=3):
        n = self.length
        chain = SingleChain(n)
        # starts with Sz = 0 subspace
        t0 = time.time()
        e, v = chain.eigen_system(k=k)
        s_tot = chain.rec_s_tot(0, n)  # the S_tot operator in m=0 subspace
        s = np.round((v.T @ s_tot @ v).diagonal())  # S_tot eigenvalues = l(l + 1)
        l = (-1 + np.sqrt(1 + 4 * s)) / 2  # finds l from s = l(l+1) which is the output ot S_tot operator
        t1 = time.time()
        print('single chain states and S_tot in Sz=0', t1 - t0)

        # the diagonal made out of EA + EB. both A and B from Sz = 0
        all_e = {0: np.add.outer(e, e).reshape(k ** 2)}

        # finds how many of each S_tot
        count = collections.Counter(l)
        # determines how many states should be considered in the subspace Sz = m
        mk = np.cumsum(list(count.values()))
        # print(mk)
        b_ab_len = np.append(k, k - mk)  # the size of (m, -m) block
        # print(b_ab_len)
        m_max = len(mk) - 1

        # building the BA - BB diagonal
        b_ab = []
        for i in range(-m_max, m_max + 1):
            b_ab = np.append(b_ab, [i] * (b_ab_len[abs(i)]) ** 2)
            # print(b_ab.shape)

        # builds |VA VB> vectors out of each pair of v
        t0 = time.time()
        print('test kron', np.kron(v, v).shape)
        print(v.shape)
        t1 = time.time()
        print('kron of the bases takes', t1 - t0)
        t0 = time.time()
        all_v = {0: np.zeros((len(v) ** 2, k, k))}
        for i in range(k):
            for j in range(k):
                all_v[0][:, i, j] = self.vector_mixer(0, 0, v[:, i], v[:, j])
        all_v[0] = np.reshape(all_v[0], (len(v) ** 2, k ** 2))
        t1 = time.time()
        print(all_v[0].shape)
        print('mixing states in Sz=0', t1 - t0)

        # building the diagonal SzSz matrix for m = 0
        t0 = time.time()
        szsz = np.concatenate(([1] * hilbert_space_size(1, n - 1) ** 2,
                               [0] * hilbert_space_size(1, n - 1) * hilbert_space_size(0, n - 1),
                               [-1] * hilbert_space_size(1, n - 1) ** 2,
                               [0] * hilbert_space_size(0, n - 1) * hilbert_space_size(0, n),
                               [-1] * hilbert_space_size(1, n - 1) ** 2,
                               [0] * hilbert_space_size(1, n - 1) * hilbert_space_size(0, n - 1),
                               [1] * hilbert_space_size(1, n - 1) ** 2))
        t1 = time.time()
        print('building the diagonal SzSz matrix for Sz = 0', t1 - t0)
        # print(szsz.shape)
        # builds S1A.S1B matrix elements between (m, -m) = (0, 0) pairs
        all_szsz = {0: np.zeros((k ** 2, k ** 2))}
        for i in range(k ** 2):
            for j in range(k ** 2):
                all_szsz[0][i, j] = szsz @ (all_v[0][:, i] * all_v[0][:, j])
                # print(x.shape)
        # print(all_szsz[0].shape)
        t1 = time.time()
        print('building effective SzSz in Sz=0', t1 - t0)

        # finds the eigen vectors for subspaces with Sz = +m and -m and builds the SzSz diagonal blocks
        for m in range(1, len(mk)):
            # print(k - mk[m - 1])
            t0 = time.time()
            ep, vp = chain.eigen_system(m=m, k=k - mk[m - 1])
            em, vm = chain.eigen_system(m=-m, k=k - mk[m - 1])
            # builds the diagonal EA + EB for SzA = m and SzB = -m and reverse
            all_e[m] = np.add.outer(em, ep).reshape((k - mk[m - 1]) ** 2)
            all_e[-m] = np.add.outer(ep, em).reshape((k - mk[m - 1]) ** 2)
            # builds |VA VB> vectors out of each pair of VA from Sz = m and VB from Sz = -m and reverse
            all_v[m] = np.zeros((len(vp) * len(vm), k - mk[m - 1], k - mk[m - 1]))
            all_v[-m] = np.zeros((len(vm) * len(vp), k - mk[m - 1], k - mk[m - 1]))
            for i in range(k - mk[m - 1]):
                for j in range(k - mk[m - 1]):
                    all_v[m][:, i, j] = self.vector_mixer(m, -m, vp[:, i], vm[:, j])
                    all_v[-m][:, i, j] = self.vector_mixer(-m, m, vm[:, i], vp[:, j])
            all_v[m] = np.reshape(all_v[m], (len(vp) * len(vm), (k - mk[m - 1]) ** 2))
            all_v[-m] = np.reshape(all_v[-m], (len(vm) * len(vp), (k - mk[m - 1]) ** 2))
            t1 = time.time()
            print('#', m, 'vectors built. took', t1 - t0)
            # building the diagonal SzSz matrix for (m, -m) pairs
            t0 = time.time()
            szszp = np.concatenate(([1] * hilbert_space_size(m + 1, n - 1) * hilbert_space_size(-m + 1, n - 1),
                                    [0] * hilbert_space_size(m + 1, n - 1) * hilbert_space_size(-m, n - 1),
                                    [-1] * hilbert_space_size(m + 1, n - 1) * hilbert_space_size(-m - 1, n - 1),
                                    [0] * hilbert_space_size(m, n - 1) * hilbert_space_size(-m, n),
                                    [-1] * hilbert_space_size(m - 1, n - 1) * hilbert_space_size(-m + 1, n - 1),
                                    [0] * hilbert_space_size(m - 1, n - 1) * hilbert_space_size(-m, n - 1),
                                    [1] * hilbert_space_size(m - 1, n - 1) * hilbert_space_size(-m - 1, n - 1)))
            szszm = np.concatenate(([1] * hilbert_space_size(-m + 1, n - 1) * hilbert_space_size(m + 1, n - 1),
                                    [0] * hilbert_space_size(-m + 1, n - 1) * hilbert_space_size(m, n - 1),
                                    [-1] * hilbert_space_size(-m + 1, n - 1) * hilbert_space_size(m - 1, n - 1),
                                    [0] * hilbert_space_size(-m, n - 1) * hilbert_space_size(m, n),
                                    [-1] * hilbert_space_size(-m - 1, n - 1) * hilbert_space_size(m + 1, n - 1),
                                    [0] * hilbert_space_size(-m - 1, n - 1) * hilbert_space_size(m, n - 1),
                                    [1] * hilbert_space_size(-m - 1, n - 1) * hilbert_space_size(m - 1, n - 1)))
            # builds S1A.S1B matrix elements between (m, -m) pairs
            all_szsz[m] = np.zeros(((k - mk[m - 1]) ** 2, (k - mk[m - 1]) ** 2))
            all_szsz[-m] = np.zeros(((k - mk[m - 1]) ** 2, (k - mk[m - 1]) ** 2))
            for i in range((k - mk[m - 1]) ** 2):
                for j in range((k - mk[m - 1]) ** 2):
                    all_szsz[m][i, j] = szszp @ (all_v[m][:, i] * all_v[m][:, j])
                    all_szsz[-m][i, j] = szszm @ (all_v[-m][:, i] * all_v[-m][:, j])
            print(all_szsz[m].shape)
            t1 = time.time()
            print('building effective SzSz in SzSz=', (m, -m), 'takes', t1 - t0)

        # orders the all_e dic from -m to m
        all_e = dict(sorted(all_e.items()))
        # builds the full diagonal EA + EB by merging all_E
        diagonal_e = np.concatenate(list(all_e.values()))

        # building S-S+ off-diagonal matrices and put the final compact S1A.S1B matrix together
        all_smsp = {}
        dim = 2 * sum(b_ab_len ** 2) - k ** 2
        sasb = np.zeros((dim, dim))
        row = 0
        col = b_ab_len[m_max] ** 2 + b_ab_len[m_max - 1] ** 2
        for m in range(-m_max, m_max):
            # building the sparse off-diagonal matrix connecting m and m + 1
            t0 = time.time()
            all_smsp[m] = np.zeros((b_ab_len[abs(m)] ** 2, b_ab_len[abs(m + 1)] ** 2))
            for i in range(b_ab_len[abs(m)] ** 2):
                for j in range(b_ab_len[abs(m + 1)] ** 2):
                    all_smsp[m][i, j] = all_v[m][:, i] @ self.smsp(m, all_v[m + 1][:, j])
                    # print(x)
            # print(all_smsp[m].shape)
            t1 = time.time()
            print('building effective S-S+ between', (m, m + 1), 'takes', t1 - t0)

            # putting the final compact S1A.S1B matrix together
            if m == -m_max:
                sasb[: b_ab_len[abs(m)] ** 2, : col] = np.concatenate((all_szsz[m], all_smsp[m]), axis=1)
            else:
                sasb[row: row + b_ab_len[abs(m)] ** 2, row - b_ab_len[abs(m - 1)] ** 2: col] = \
                    np.concatenate((all_smsp[m - 1].T, all_szsz[m], all_smsp[m]), axis=1)
            row = row + b_ab_len[abs(m)] ** 2
            col = col + b_ab_len[abs(m + 2)] ** 2
            # print('row', row)
            # print('column', col)
        sasb[row:, row - b_ab_len[abs(m_max - 1)] ** 2:] = \
            np.concatenate((all_smsp[m_max - 1].T, all_szsz[m_max]), axis=1)  # adds the last row

        return diagonal_e, b_ab, sasb, 2 * sum(b_ab_len) - k

    # finds the first k2 eigen values and eigen states of the effective two chain Hamiltonian(k1),
    # with BA - BB = bab, and interchain coupling jab.
    # also spits out the total number of single chain states used.
    @lru_cache  # this memoize
    def eigen_system_eff(self, bab=0, jab=0, m=0, k1=3, k2=6, vectors=True):
        n = self.length
        if n >= 8:
            try:
                e_ab = np.loadtxt('data/pairs/Diagonal_AB_' + str(n) + '-' + str(k1) + '.dat')
                sasb = np.loadtxt('data/pairs/SASB_' + str(n) + '-' + str(k1) + '.dat')
                b_ab = np.loadtxt('data/pairs/B_AB_' + str(n) + '-' + str(k1) + '.dat')
                single_states = np.loadtxt('data/pairs/States-Count_' + str(n) + '-' + str(k1) + '.dat')
                # print(single_states, b_ab.shape, e_ab.shape, sasb.shape)
            except IOError:
                print('saved data was not available')
                e_ab, b_ab, sasb, single_states = self.hamiltonian_eff_sz(m=m, k=k1)
                np.savetxt('data/pairs/Diagonal_AB_' + str(n) + '-' + str(k1) + '.dat', e_ab)
                np.savetxt('data/pairs/SASB_' + str(n) + '-' + str(k1) + '.dat', sasb)
                np.savetxt('data/pairs/B_AB_' + str(n) + '-' + str(k1) + '.dat', b_ab)
                np.savetxt('data/pairs/States-Count_' + str(n) + '-' + str(k1) + '.dat', single_states)
                # print(single_states, b_ab.shape, e_ab.shape, sasb.shape)
        else:
            e_ab, b_ab, sasb, single_states = self.hamiltonian_eff_sz(m=m, k=k1)
            # print(single_states, b_ab.shape, e_ab.shape, sasb.shape)
        h = diags(e_ab) + bab * diags(b_ab) + jab * sasb
        return linalg.eigsh(h, k=k2, which='SA', return_eigenvectors=vectors), single_states

    # finds the first k2 eigen values and eigen states of the effective two chain Hamiltonian(k1),
    # when BA - BB -> infinity.
    @lru_cache  # this memoize
    def eigen_system_eff_pol(self, jab=0, k1=3, k2=6, vectors=True):
        n = self.length
        if n >= 8:
            try:
                e_ab = np.loadtxt('data/pairs/Diagonal_AB_' + str(n) + '-' + str(k1) + '.dat')
                sasb = np.loadtxt('data/pairs/SASB_' + str(n) + '-' + str(k1) + '.dat')
                b_ab = np.loadtxt('data/pairs/B_AB_' + str(n) + '-' + str(k1) + '.dat')
                single_states = np.loadtxt('data/pairs/States-Count_' + str(n) + '-' + str(k1) + '.dat')
                # print(single_states, b_ab.shape, e_ab.shape, sasb.shape)
            except IOError:
                print('saved data was not available')
                e_ab, b_ab, sasb, single_states = self.hamiltonian_eff_sz(k=k1)
                np.savetxt('data/pairs/Diagonal_AB_' + str(n) + '-' + str(k1) + '.dat', e_ab)
                np.savetxt('data/pairs/SASB_' + str(n) + '-' + str(k1) + '.dat', sasb)
                np.savetxt('data/pairs/B_AB_' + str(n) + '-' + str(k1) + '.dat', b_ab)
                np.savetxt('data/pairs/States-Count_' + str(n) + '-' + str(k1) + '.dat', single_states)
                # print(single_states, b_ab.shape, e_ab.shape, sasb.shape)
        else:
            e_ab, b_ab, sasb, single_states = self.hamiltonian_eff_sz(k=k1)
            # print(single_states, b_ab.shape, e_ab.shape, sasb.shape)
        single_states = single_states[:, 0]
        mid = int(len(single_states) / 2)
        idx0 = int((single_states[: mid] ** 2).sum())
        # print(idx0)
        e_ab = e_ab[idx0: idx0 + k1 ** 2]
        sasb = sasb[idx0: idx0 + k1 ** 2, idx0: idx0 + k1 ** 2]
        # print(e_ab.shape, sasb.shape)
        h = diags(e_ab) + jab * sasb
        return linalg.eigsh(h, k=k2, which='SA', return_eigenvectors=vectors)


if __name__ == "__main__":

    def chop(x):
        if abs(x) > 1e-12:
            return x
        else:
            return 0


    Chop = np.vectorize(chop)

    L = 8  # length for test
    M = 0  # the total angular momentum
    # print('for two chains of length', L, 'the size of S_tot^z = 0 subspace size is', hilbert_space_size(M, 2 * L))
    print('for a chain of length', L, 'the size of S_tot^z =', M, 'subspace size is', hilbert_space_size(M, L))

    Chains = TwoChains(L)
    K1 = 4
    start = time.time()
    Diagonal_E, B_ab, SaSb, States = Chains.hamiltonian_eff(k=K1)
    end = time.time()
    # np.savetxt('/tmp/SASB.dat', SaSb)
    # np.savetxt('/tmp/B_AB.dat', B_ab)
    # np.savetxt('/tmp/Diagonal_AB.dat', Diagonal_E)

    # SASB = np.loadtxt('data/from_hercules/SASB_8-14-44.dat')
    # print(sum(sum(SASB - SaSb)))

    # Bab = 0.1
    # Jab = 0.2

    # E, V, States = Chains.eigen_system(Bab, Jab, k1=K1)

    print('it took', end - start, 'seconds to build the effective two chain Hamiltonian')
    print('number of single chain states used:', States)
    # # print(E)
    print('matrix size', len(B_ab))
    # np.savetxt('/tmp/SASB.dat', SS)
    # print(SS.shape)
    # print(Diagonal_E.shape)
    # print(max(B_AB))

    # Chain = SingleChain(L)
    # start = time.time()
    # H = Chain.rec_chain_hamiltonian_szm(0, L)
    # B = 0
    # Hb = B * Chain.local_field(0)
    # end = time.time()
    # print('number of nonzero elements =', H.nnz)
    # print('it takes', (end - start), 'seconds to build the matrix')
    #
    # start = time.time()
    # E0, V0 = Chain.eigen_system(M, k=6)
    # end = time.time()
    # print('it takes', (end - start), 'seconds to find the eigenvalues')
    # print('Delta =', E0[1] - E0[0])
    # print('Gamma =', E0[2] - E0[1])

    # start = time.time()
    # EF0, VF0 = Chain.eigen_system_full(b0=1e-8, k=10)
    # # EF, VF = Chain.eigen_system_full(b1=0.1)
    # end = time.time()
    # print('it takes', (end - start), 'seconds to find the eigenstates in the full Hilbert space')
    # Max_amp = np.max(VF0 ** 2, axis=1)
    # print(Max_amp[Max_amp > 1e-3].shape)
    # print(3**L)
    # print(hilbert_space_size(2, L))
    # print('S_tot^z =', Chain.sz_tot_full(VF0))

    # start = time.time()
    # C = Chain.configs(0)
    # end = time.time()
    # print('it takes', (end - start), 'seconds to build the configurations')
    # # print(C)
    #
    # start = time.time()
    # CR = Chain.rec_configs(0, L)
    # end = time.time()
    # print('it takes', (end - start), 'seconds to build the configurations recursively')
    # # print(CR.toarray())
    # print(sum(CR - C))

    # start = time.time()
    # B = 0.3
    # T = np.linspace(0, 10, 15)
    # a0 = np.zeros((len(T), 2))
    # a1 = np.zeros((len(T), 2))
    # for j in range(len(T)):
    #     U = Chain.unitary(B, T[j])
    #     a0[j, :] = abs(V0[:, :2].T @ U @ V0[:, 0]) ** 2
    #     a1[j, :] = abs(V0[:, :2].T @ U @ V0[:, 1]) ** 2
    # end = time.time()
    # print('it takes', (end - start), 'seconds to compute the unitary evolution')
    # start = time.time()
    # A0 = np.zeros((len(T), 2))
    # A1 = np.zeros((len(T), 2))
    # for j in range(len(T)):
    #     UA = Chain.unitary_approx(B, T[j], d=10)
    #     A0[j, :] = np.abs([UA[0, 0], UA[1, 0]]) ** 2
    #     A1[j, :] = np.abs([UA[0, 1], UA[1, 1]]) ** 2
    # end = time.time()
    # print('it takes', (end - start), 'seconds to compute the approx unitary evolution')
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # ax1.plot(T, a0[:, 0])
    # ax1.plot(T, a0[:, 1])
    # ax1.plot(T, 1 - a0[:, 0] - a0[:, 1])
    # ax1.plot(T, A0[:, 0], ':')
    # ax1.plot(T, A0[:, 1], '.')
    # ax1.plot(T, 1 - A0[:, 0] - A0[:, 1], '.')
    # ax1.axhline(y=0.5, ls='--', color='r')
    # ax2.plot(T, a1[:, 0])
    # ax2.plot(T, a1[:, 1])
    # ax2.plot(T, 1 - a1[:, 0] - a1[:, 1])
    # ax2.plot(T, A1[:, 0], ':')
    # ax2.plot(T, A1[:, 1], '.')
    # ax2.plot(T, 1 - A1[:, 0] - A1[:, 1], '.')
    # ax2.axhline(y=0.5, ls='--', color='r')
    # plt.show()

    # start = time.time()
    # B = np.linspace(0, 0.25, 15)
    # eps = np.zeros((len(B), 2))
    # for j in range(len(B)):
    #     eps[j, :] = Chain.admix(B[j])
    # end = time.time()
    # print('it takes', (end - start), 'seconds to find the admixture')
    # # print('(eps0, eps1) =', eps.shape)
    # plt.figure()
    # plt.plot(B, eps)
    # plt.axhline(y=0.01, ls='--', color='grey')
    # plt.axvline(x=E0[1]-E0[0], ls='--', color='k')
    #
    #
    # start = time.time()
    # SZ = Chain.sz_avg(0, (V0[:, 0] - V0[:, 1])/np.sqrt(2))
    # end = time.time()
    # print('it takes', (end - start), 'seconds to compute <Sz>')
    # # print(SZ)
    # plt.figure()
    # plt.bar(range(1, L+1), SZ)
    # plt.axhline(y=-0.5, ls='--', color='grey')
    # plt.axhline(y=0.5, ls='--', color='grey')
    # plt.show()

    # start = time.time()
    # S = Chain.rec_s_tot(M, L)
    # end = time.time()
    # print('it takes', (end - start), 'seconds to build the S_tot matrix')
    # print(np.round((V0.T @ S @ V0).diagonal()))
    #
    # start = time.time()
    # SF = Chain.rec_s_tot_full()
    # end = time.time()
    # print('it takes', (end - start), 'seconds to build the S_tot matrix in the full Hilbert space')
    # print(np.round((VF0.T @ SF @ VF0).diagonal()))
