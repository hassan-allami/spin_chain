import numpy as np
from Haldane_chain_qubit import *

L = 8  # length for test
M = 0  # the total angular momentum
print('for two chains of length', L, 'the size of S_tot^z = 0 subspace size is', hilbert_space_size(M, 2 * L))
print('for a chain of length', L, 'the size of S_tot^z =', M, 'subspace size is', hilbert_space_size(M, L))
Chain = SingleChain(2 * L)
Chains = TwoChains(L)
K1 = 10

start = time.time()
Diagonal_E_z, B_ab_z, SaSb_z, States = Chains.hamiltonian_eff_sz(k=K1)
end = time.time()
print('it took', end - start, 'seconds to build the effective two chain Hamiltonian')
print('number of single chain states used:', States)
print('matrix size', len(B_ab_z))
print('\n')

start = time.time()
Diagonal_E, B_ab, SaSb, States = Chains.hamiltonian_eff(k=K1)
end = time.time()

# np.savetxt('SASB.dat', SaSb)
# np.savetxt('B_AB.dat', B_ab)
# np.savetxt('Diagonal_AB.dat', Diagonal_E)

print('it took', end - start, 'seconds to build the effective two chain Hamiltonian')
print('number of single chain states used:', States)
print('matrix size', len(B_ab))
print('\n')

# start = time.time()
# Diagonal_E_2, B_ab_2, SaSb_2, States = Chains.hamiltonian_eff_2(k=K1)
# end = time.time()
# print('it took', end - start, 'seconds to build the effective two chain Hamiltonian')
# print('number of single chain states used:', States)
# print('matrix size', len(B_ab))
# print('\n')


# print('check:', ((Diagonal_E_2 - Diagonal_E)**2).sum())
# print('check:', ((B_ab_2 - B_ab)**2).sum())
# print('check:', ((SaSb_2.diagonal() - SaSb.diagonal())**2).sum())
# print('check:', ((abs(SaSb_2) - abs(SaSb))**2).sum())


# H20 = np.diag(Diagonal_E) + SaSb
# H22 = np.diag(Diagonal_E) + SaSb
# E20 = linalg.eigsh(H20, k=3, which='SA', return_eigenvectors=False)
# E22 = linalg.eigsh(H22, k=3, which='SA', return_eigenvectors=False)
# E2L = Chain.eigen_system(vectors=False)
# print('E20=', E20)
# print('E22=', E22)
# print('E2L=', E2L)

