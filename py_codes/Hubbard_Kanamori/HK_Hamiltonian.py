import numpy as np
import scipy.linalg
from scipy import sparse as spr
from scipy.special import comb
from itertools import product
from functools import lru_cache


class HubbardKanamori:
    # length is the number of dots in the cain
    # sz_tot is the Hilbert subspace to work in
    # if all_sz True then doesn't use sz_tot restriction
    # the default total number of particles is 2 * length
    # use custom_n_tot to choose custom total number of particles
    # if all_n is True doesn't use n_tot restriction
    def __init__(self, length: int,
                 custom_n_tot=False, all_n: bool = False,
                 sz_tot: int = 0, all_sz: bool = False):
        self.length = length
        self.sz_tot = sz_tot  # could be between -length to +length
        self.all_sz = all_sz
        self.all_n = all_n
        if custom_n_tot is False:
            self.n_tot = 2 * length  # the default number of particles
        else:
            self.n_tot = custom_n_tot

    # dimension of the Hilbert subspace
    # note that not all n_tot and sz_tot are compatible
    @lru_cache  # this memoize
    def hilbert_dimension(self):
        if self.all_n:
            n = None
        else:
            n = self.n_tot
        if n is None:
            if self.all_sz:
                return int(16 ** self.length)
            else:
                # this is based on a neat combinatorics identity
                return int(comb(4 * self.length, 2 * self.length + 2 * self.sz_tot))
        else:
            if self.all_sz:
                return int(comb(4 * self.length, n))
            else:
                return int(comb(2 * self.length, n / 2 + self.sz_tot) * comb(2 * self.length, n / 2 - self.sz_tot))

    # gives the dictionary of configurations
    # the first 2N orbitals are up and the second 2N are down
    # the even orbitals are p+ and the odd orbitals are p-
    @lru_cache  # this memoize
    def configs(self):
        confs = list(product([0, 1], repeat=4 * self.length))
        if self.all_n:
            n = None
        else:
            n = self.n_tot

        if n is None:
            if self.all_sz:
                pass
            else:
                confs = [tup for tup in confs if \
                         sum(tup[: 2 * self.length]) - sum(tup[2 * self.length:]) == 2 * self.sz_tot]
        else:
            if self.all_sz:
                confs = [tup for tup in confs if sum(tup) == n]
            else:
                confs = [tup for tup in confs if sum(tup) == n]
                confs = [tup for tup in confs if \
                         sum(tup[: 2 * self.length]) - sum(tup[2 * self.length:]) == 2 * self.sz_tot]

        return {tpl: idx for idx, tpl in enumerate(confs)}

    # creates n_{i, sigma} operator
    # where i is the orbital number not the site number
    # spin up is True and spin down is False
    @lru_cache  # this memoize
    def n_i_sigma(self, orbit: int, spin: bool):
        confs = self.configs()
        dim = len(confs)
        dat = []
        row = []
        col = []
        for c in confs:
            if spin:
                if c[orbit] == 1:
                    # print(c)
                    dat.append(1)
                    row.append(confs[c])
                    col.append(confs[c])
            else:
                if c[orbit + 2 * self.length] == 1:
                    # print(c)
                    dat.append(1)
                    row.append(confs[c])
                    col.append(confs[c])
        return spr.coo_matrix((dat, (row, col)), shape=(dim, dim))

    # creates rho_i = n_{i, up} + n_{i, down}
    # where i is the orbital number not the site number
    # i is between 0 to 2N - 1, where N is the number of dots
    @lru_cache  # this memoize
    def rho_i(self, orbit: int):
        return self.n_i_sigma(orbit, spin=True) + self.n_i_sigma(orbit, spin=False)

    # creates the density operator at a given dot
    # where rho_site = rho_{site, p+} + rho_{site, p+}
    # where site is the site number not the orbital number
    # site is between 0 to N-1, where N is the number of dots
    @lru_cache  # this memoize
    def rho_site(self, site: int):
        return self.rho_i(2 * site) + self.rho_i(2 * site + 1)

    # creates hopping operator c+i * cj (from the second to the first) and the same spin
    # where i and j are orbital numbers not site numbers
    # spin up is True and spin down is False
    # i and j are between 0 to 2N - 1, where N is the number of dots
    @lru_cache  # this memoize
    def hop_ij_sigma(self, orbits: (int, int), spin: bool):
        if orbits[0] == orbits[1]:
            return self.n_i_sigma(orbit=orbits[0], spin=spin)
        elif orbits[0] < orbits[1]:
            confs = self.configs()
            dim = len(confs)
            dat = []
            row = []
            col = []
            for c in confs:
                if spin:
                    if c[orbits[0]] == 0 and c[orbits[1]] == 1:
                        # print('from', c)
                        phase = sum(c[orbits[0]: orbits[1]]) % 2
                        new_c = list(c)
                        new_c[orbits[0]] = 1
                        new_c[orbits[1]] = 0
                        new_c = tuple(new_c)
                        # print('to', new_c)
                        dat.append((-1) ** phase)
                        row.append(confs[new_c])
                        col.append(confs[c])
                else:
                    if c[orbits[0] + 2 * self.length] == 0 and c[orbits[1] + 2 * self.length] == 1:
                        # print('from', c)
                        phase = sum(c[orbits[0] + 2 * self.length: orbits[1] + 2 * self.length]) % 2
                        new_c = list(c)
                        new_c[orbits[0] + 2 * self.length] = 1
                        new_c[orbits[1] + 2 * self.length] = 0
                        new_c = tuple(new_c)
                        # print('to', new_c)
                        dat.append((-1) ** phase)
                        row.append(confs[new_c])
                        col.append(confs[c])
            return spr.coo_matrix((dat, (row, col)), shape=(dim, dim))
        else:
            return self.hop_ij_sigma(orbits=orbits[::-1], spin=spin).T

    # creates intra-dot hopping from p- to p+ at given site summed over both spins
    # site is between 0 to N-1, where N is the number of dots
    # it's the operator L+
    @lru_cache  # this memoize
    def lp_site(self, site: int):
        mat = self.hop_ij_sigma(orbits=(2 * site, 2 * site + 1), spin=True)
        mat += self.hop_ij_sigma(orbits=(2 * site, 2 * site + 1), spin=False)
        return mat

    # creates intra-dot hopping from p+ to p- at given site summed over both spins
    # site is between 0 to N-1, where N is the number of dots
    # it's L- the conjugate of lp_site = L+
    @lru_cache  # this memoize
    def lm_site(self, site: int):
        return self.lp_site(site=site).T

    # creates intra-dot hopping between p+ and p- at given site summed over both spins
    # site is between 0 to N-1, where N is the number of dots
    # it's lp_site + lm_site
    @lru_cache  # this memoize
    def intra_hop(self, site: int):
        return self.lp_site(site=site) + self.lm_site(site=site)

    # creates L+_i * L-_j + h.c. between sites i and j
    # site number is between 0 to N-1, where N is the number of dots
    @lru_cache  # this memoize
    def inter_ell(self, sites: (int, int)):
        mat = self.lp_site(sites[0]) @ self.lm_site(sites[1])
        return mat + mat.T

    # creates inter-dot hopping sites i and j summed over both spins and internal orbitals
    # it's only between the same orbitals and the same spins
    # site number is between 0 to N-1, where N is the number of dots
    @lru_cache  # this memoize
    def inter_hop(self, sites: (int, int)):
        mat = self.hop_ij_sigma(orbits=(2 * sites[0], 2 * sites[1]), spin=True)
        mat += self.hop_ij_sigma(orbits=(2 * sites[0], 2 * sites[1]), spin=True).T
        mat += self.hop_ij_sigma(orbits=(2 * sites[0] + 1, 2 * sites[1] + 1), spin=True)
        mat += self.hop_ij_sigma(orbits=(2 * sites[0] + 1, 2 * sites[1] + 1), spin=True).T
        mat += self.hop_ij_sigma(orbits=(2 * sites[0], 2 * sites[1]), spin=False)
        mat += self.hop_ij_sigma(orbits=(2 * sites[0], 2 * sites[1]), spin=False).T
        mat += self.hop_ij_sigma(orbits=(2 * sites[0] + 1, 2 * sites[1] + 1), spin=False)
        mat += self.hop_ij_sigma(orbits=(2 * sites[0] + 1, 2 * sites[1] + 1), spin=False).T
        return mat

    # creates sz_i * sz_j
    # where sz_i = (n_{i, up} - n_{i, down})/2
    # and i and j are orbital numbers not site numbers
    @lru_cache  # this memoize
    def szi_szj(self, orbits: (int, int)):
        sz0 = (self.n_i_sigma(orbits[0], spin=True) -
               self.n_i_sigma(orbits[0], spin=False)) / 2
        sz1 = (self.n_i_sigma(orbits[1], spin=True) -
               self.n_i_sigma(orbits[1], spin=False)) / 2
        return sz0 @ sz1

    # creates s+_i * s-_j
    # where s+_i = ci+_up * ci_down
    # and i and j are orbital numbers not site numbers
    # i and j must be different
    @lru_cache  # this memoize
    def spi_smj(self, orbits: (int, int)):
        if orbits[0] < orbits[1]:
            confs = self.configs()
            dim = len(confs)
            dat = []
            row = []
            col = []
            for c in confs:
                if c[orbits[0]] == 0 and c[orbits[1]] == 1 and \
                        c[orbits[0] + 2 * self.length] == 1 and \
                        c[orbits[1] + 2 * self.length] == 0:
                    # print('from:', c)
                    phase = (sum(c[orbits[0]: orbits[1]]) +
                             sum(c[orbits[0] + 2 * self.length: \
                                   orbits[1] + 2 * self.length])) % 2
                    new_c = list(c)
                    new_c[orbits[0]] = 1
                    new_c[orbits[1]] = 0
                    new_c[orbits[0] + 2 * self.length] = 0
                    new_c[orbits[1] + 2 * self.length] = 1
                    new_c = tuple(new_c)
                    # print('to  :', new_c)
                    dat.append((-1) ** phase)
                    row.append(confs[new_c])
                    col.append(confs[c])
            return spr.coo_matrix((dat, (row, col)), shape=(dim, dim))
        elif orbits[0] > orbits[1]:
            return self.spi_smj(orbits[::-1]).T
        else:
            print('i and j should be different')

    # creates s_i . s_j
    # where i and j are orbital numbers not site numbers
    @lru_cache  # this memoize
    def si_sj(self, orbits: (int, int)):
        if orbits[0] == orbits[1]:
            return 3 * self.szi_szj(orbits)
        else:
            return (self.szi_szj(orbits) +
                    (self.spi_smj(orbits) + self.spi_smj(orbits[::-1])) / 2)

    # creates the total angular momentum operator:
    # S_tot = (sum_i[s_i])**2, where i is the orbital index
    @lru_cache()  # this memoize
    def s_tot(self):
        # initiate an empty sparse matrix with the right dimension
        dim = self.hilbert_dimension()
        s_mat = spr.coo_matrix((dim, dim))
        # orbit index is i
        for i in range(2 * self.length):
            s_mat += self.si_sj((i, i))
            for j in range(i):
                s_mat += 2 * self.si_sj((i, j))
        return s_mat

    # this creates the U term:
    # sum_i[(rho_i+ - 1) * (rho_i- - 1) + n_i+up*n_i+down + n_i-up*n_i-down]
    @lru_cache  # this memoize
    def u_term(self):
        # the constant term from sum_i[(rho_i+ - 1) * (rho_i- - 1)]
        u_mat = spr.diags([self.length] * self.hilbert_dimension())
        # 2i is the index of p+ at site i and
        # 2i+1 is the index of p - at site i
        for i in range(self.length):
            # rho+ * rho- - (rho+ + rho-)
            rho_term = self.rho_i(2*i) @ self.rho_i(2*i + 1)
            rho_term -= self.rho_i(2*i) + self.rho_i(2*i + 1)
            # the hubbard terms
            n_term = self.n_i_sigma(2*i, True) @ self.n_i_sigma(2*i, False)
            n_term += self.n_i_sigma(2*i+1, True) @ self.n_i_sigma(2*i+1, False)
            # add both to the u_mat
            u_mat += rho_term + n_term
        return u_mat

    # this creates the exchange term:
    # sum_i[2*s_i+ . si_- + rho_i+ * rho_i- / 2]
    @lru_cache()  # this memoize
    def w_term(self):
        # initiate an empty sparse matrix with the right dimension
        dim = self.hilbert_dimension()
        w_mat = spr.coo_matrix((dim, dim))
        # 2i is the index of p+ at site i and
        # 2i+1 is the index of p - at site i
        for i in range(self.length):
            w_mat += 2 * self.si_sj((2*i, 2*i+1))
            w_mat += self.rho_i(2*i) @ self.rho_i(2*i + 1) / 2
        return w_mat

    # this creates the intra hopping term:
    # sum_i_sigma[c+_i+sigma c_i-sigma + h.c.]
    @lru_cache()  # this memoize
    def delta_term(self):
        # initiate an empty sparse matrix with the right dimension
        dim = self.hilbert_dimension()
        delta_mat = spr.coo_matrix((dim, dim))
        # site index is i
        for i in range(self.length):
            delta_mat += self.intra_hop(i)
        return delta_mat

    # this creates the inter-dot hopping:
    # sum_i_a_sigma[c+_i_a_sigma c_i+1_a_sigma + h.c.]
    @lru_cache()  # this memoize
    def t_term(self):
        # initiate an empty sparse matrix with the right dimension
        dim = self.hilbert_dimension()
        t_mat = spr.coo_matrix((dim, dim))
        # site index is i
        for i in range(self.length - 1):
            t_mat += self.inter_hop((i, i+1))
        return t_mat

    # this creates the V term:
    # sum_i[(rho_i - 2) * (rho_i+1 - 2)]
    @lru_cache  # this memoize
    def v_term(self):
        # the constant term
        v_mat = spr.diags([4 * (self.length - 1)] * self.hilbert_dimension())
        # site index is i
        for i in range(self.length - 1):
            # rho_i * rho_i+1 - 2*(rho_i + rho_i+1)
            v_mat += self.rho_site(i) @ self.rho_site(i+1)
            v_mat -= 2 * self.rho_site(i) + self.rho_site(i+1)
        return v_mat

    # this creates the Gamma term:
    # sum_i[L^+_i * L^-_i+1 + h.c.]
    @lru_cache()  # this memoize
    def gamma_term(self):
        # initiate an empty sparse matrix with the right dimension
        dim = self.hilbert_dimension()
        gamma_mat = spr.coo_matrix((dim, dim))
        # site index is i
        for i in range(self.length - 1):
            gamma_mat += self.inter_ell((i, i+1))
        return gamma_mat

    # puts all terms together and makes the Hubbard-Kanamori Hamiltonian
    # I set the default value of (t, Delta, Gamma) = (1, 0, 0)
    @lru_cache()  # this memoize
    def ham(self, u, w, v, t=1, delta=0, gamma=0):
        return (u * self.u_term() - w * self.w_term() +
                0.5*delta * self.delta_term() +
                t * self.t_term() + v * self.v_term() +
                gamma * self.gamma_term())


if __name__ == "__main__":
    N = 4
    Sz_tot = 0
    N_tot = 2
    HK = HubbardKanamori(length=N, custom_n_tot=False, all_n=False,
                         sz_tot=Sz_tot, all_sz=False)

    print(HK.hilbert_dimension())
    print('\n'.join(f'{key}: {value}' for key, value in HK.configs().items()))
    print(len(HK.configs()))

    # Orbital = 1
    # Dot = 1
    # Orbit_Pair = (0, 1)
    # Dot_Pair = (1, 0)
    # print('n_up \n', HK.n_i_sigma(Orbital, True))
    # print('n_down \n', HK.n_i_sigma(Orbital, False))
    # print('rho_i \n', HK.rho_i(Orbital))
    # print('rho_i \n', HK.rho_i(Orbital - 1))
    # print('rho_site \n', HK.rho_site(Dot))

    # print('hop \n', HK.hop_ij_sigma(orbits=Orbit_Pair, spin=False))
    # print('hop \n', HK.hop_ij_sigma(orbits=Orbit_Pair[::-1], spin=False))
    # print('Lp: \n', HK.lp_site(site=Dot))
    # print('Lm: \n', HK.lm_site(site=Dot))
    # print('intra_hop: \n', HK.intra_hop(site=Dot))
    # print('inter_hop: \n', HK.inter_hop(sites=Dot_Pair))

    # print('sz_i * sz_j: \n', HK.szi_szj(Orbit_Pair))
    # print('sp_i * sm_j: \n', HK.spi_smj(Orbit_Pair))
    # print('s_i * s_j: \n', HK.si_sj(Orbit_Pair))
    # print('S_tot: \n', HK.s_tot())

    # print('U term: \n', HK.u_term())
    # print('W term: \n', HK.w_term())
    # print('Delta term: \n', HK.delta_term())
    # print('t term: \n', HK.t_term())
    # print('V term: \n', HK.v_term())
    # print('Gamma term: \n', HK.gamma_term())

    U = 3.515
    W = 10
    V = 2

    H = HK.ham(U, W, V, t=1, delta=0, gamma=0)
    print(H.shape)
    S = HK.s_tot()
    K = 3
    E, V = spr.linalg.eigsh(H, k=K, which='SA')
    print('E =', E.real.round(3))
    G = np.diff(E)
    print('gap ratio=', (G[0] / G[1]).round(3))
    print((V.T @ S @ V).round(2))

    print('Hilbert Space Size for N = 8:', HubbardKanamori(8).hilbert_dimension())

