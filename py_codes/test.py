import time
import numpy as np
from Haldane_chain_qubit import *
from matplotlib import pyplot as plt

L = 8
K1 = 10
K2 = 10
M = 0
Jab = 0
Bab = 0

chain = SingleChain(2 * L)
Pairs = TwoChains(L)

E, V = Pairs.eigen_system_eff_pol(Jab, K1, K2)
print(E)
print(V.shape)
print(np.round(V[:, 3]))

# Sys, States = Pairs.eigen_system_eff(Bab, Jab, M, K1, K2)
# # print(States)
#
# Base_index = compute_basis(States)
# print(Base_index)
#
# V = Sys[1]
# print(V.shape)
# print([idx for idx, v in enumerate(V[:, 0]) if abs(np.round(1*v)) == 1])

# chain = SingleChain(L)
# e, v = chain.eigen_system(k=K1)
# s_tot = chain.rec_s_tot(0, L)  # the S_tot operator in m=0 subspace
# s = np.round((v.T @ s_tot @ v).diagonal())  # S_tot eigenvalues = l(l + 1)
# l = (-1 + np.sqrt(1 + 4 * s)) / 2  # finds l from s = l(l+1) which is the output ot S_tot operator
# t1 = time.time()
# # print('single chain states and S_tot in Sz=0', t1 - t0)
# # finds how many of each S_tot
# count = collections.Counter(l)
# # determining how many states should be considered in the subspace Sz = m
# mk = np.cumsum(list(count.values()))
# # print(l)
# # print(mk)
# # the number of the considered single chain states with S_tot^z = m
# m_len = np.append(K1, K1 - mk)
# # print(m_len)
# m_max = len(mk) - 1
# # print(m_max)
# if M > 0:
#     ms = range(max(-m_max, m - m_max), m_max + 1)
#     print(np.asarray(ms))
#     # print(m - np.asarray(ms))
# else:
#     ms = range(-m_max, min(m_max, M + m_max) + 1)
#     print(np.asarray(ms))
#     # print(m - np.asarray(ms))
# states = []
# for ma in ms:
#     # counting the number of single chains states used
#     states = np.append(states, [m_len[abs(ma)], m_len[abs(M-ma)]])
#
# states = states.reshape((len(ms), 2))
# print(states)
# print(sum(states))
#
# np.savetxt('data/pairs/States-Count_' + str(L) + '-' + str(K1) + '.dat', states)

# start = time.time()
# H0, B_ab0, SaSb0 = chains.hamiltonian_full()
# end = time.time()
# print('Sz=0 full:', end - start)
# E0 = linalg.eigsh(H0 + Jab * SaSb0, return_eigenvectors=False)

# start = time.time()
# H0, B_ab0, SaSb0 = chains.hamiltonian_full_sz(m=M)
# end = time.time()
# print('general full building matrix:', end - start)
# start = time.time()
# E0 = linalg.eigsh(H0 + Bab * B_ab0 + Jab * SaSb0, return_eigenvectors=False, which='SA')
# end = time.time()
# print('general full finding spectrum:', end - start)
# print('matrix size', H0.shape, '\n')
# print(E0)

# start = time.time()
# H = chain.rec_chain_hamiltonian_szm(m=M, n=2*L)
# E = linalg.eigsh(H, return_eigenvectors=False)
# end = time.time()
# print('single double chain time:', end - start, '\n')
# print(E)

# start = time.time()
# H_f, B_ab_f, SaSb_f, States = chains.hamiltonian_eff_sz(m=M, k=K1)
# end = time.time()
# print('general effective building matrix:', end - start)
# start = time.time()
# E_f = linalg.eigsh(diags(H_f) + Bab * diags(B_ab_f) + Jab * SaSb_f, return_eigenvectors=False, which='SA')
# end = time.time()
# print('general effective finding spectrum:', end - start)
# print('single chain states:', States)
# print('matrix size:', SaSb_f.shape, '\n')
# print(E_f)

# print('check:', np.sqrt(sum((E - E0)**2))/len(E0))
# print('check eff:', np.sqrt(sum((E_f - E0)**2))/len(E0))
# print((H - H0 - SaSb0).nnz)

# print('\n', 'Delta =', E0[-2] - E0[-1])

# t = np.loadtxt('/home/hassan/Downloads/menaf_tmp/time.txt')
# u00 = np.loadtxt('/home/hassan/Downloads/menaf_tmp/U_values_t_squared00.txt')
# u01 = np.loadtxt('/home/hassan/Downloads/menaf_tmp/U_values_t_squared01.txt')
# u10 = np.loadtxt('/home/hassan/Downloads/menaf_tmp/U_values_t_squared10.txt')
# u11 = np.loadtxt('/home/hassan/Downloads/menaf_tmp/U_values_t_squared11.txt')
#
# ch = SingleChain(L)
# E, V = ch.eigen_system(k=2)
# B1 = E[1] - E[0]
# print(len(V))
# D = 10
# U = np.zeros((len(t), D, D))
# U_F = np.empty((len(t), len(V), len(V)), dtype=complex)
# print(U_F.shape)
# for i in range(len(t)):
#     U[i, :, :] = abs(ch.unitary_approx(B1, t[i], D)) ** 2
#     # print(ch.unitary(B1, t[i])[:3, :3])
#     # print(U_F[i, :, :].shape)
#     U_F[i, :, :] = ch.unitary(B1, t[i])
#     print(i)

# U_V = V.T @ U_F @ V
# print(U_V.shape)
# print(U_V[2])
# U_V = abs(U_V)**2
# print(U_V[2])

# print('u00 check', ((u00 - U[:, 0, 0]) ** 2).sum())
# print('u01 check', ((u01 - U[:, 0, 1]) ** 2).sum())
# print('u10 check', ((u10 - U[:, 1, 0]) ** 2).sum())
# print('u11 check', ((u11 - U[:, 1, 1]) ** 2).sum())
# print('\n')
# print('u00 check', ((u00 - U_V[:, 0, 0]) ** 2).sum())
# print('u01 check', ((u01 - U_V[:, 0, 1]) ** 2).sum())
# print('u10 check', ((u10 - U_V[:, 1, 0]) ** 2).sum())
# print('u11 check', ((u11 - U_V[:, 1, 1]) ** 2).sum())
# print('\n')
#
# r0 = 1 - u00 - u01
# r1 = 1 - u10 - u11
# R0 = 1 - U[:, 0, 0] - U[:, 0, 1]
# R1 = 1 - U[:, 1, 0] - U[:, 1, 1]
# RF0 = 1 - U_V[:, 0, 0] - U_V[:, 0, 1]
# RF1 = 1 - U_V[:, 1, 0] - U_V[:, 1, 1]
# print(r0 - R0, '\n')
# print(RF0 - R0, '\n')
# print(r0 - RF0)
# print(r1 - R1)

# E_diagonal, B_ab, SaSb, States = chains.hamiltonian_eff_sz(m=M, k=K1)
# print('matrix size:', B_ab.shape)
# print('number of single chain states:', States)
# E = linalg.eigsh(np.diag(E_diagonal) + Jab * SaSb, return_eigenvectors=False)
#
# E_2 = chain.eigen_system(m=M, k=6, vectors=False)
# print(E_2)
# print((E - E_2)**2)


# Levels = chains.uncoupled_spect(k=K1)
# print((np.subtract(sorted(Levels[Levels[:, 1] == M][:, 0]), sorted(E_diagonal))**2).sum())


#
# i = 1
# while i < hilbert_space_size(m, n - 1) + hilbert_space_size(m + 1, n - 1):
#     d = block_diag((
#         diags([1] * (hilbert_space_size(m, n - 1) + hilbert_space_size(m + 1, n - 1))),
#         coo_matrix((hilbert_space_size(m - 1, n - 1), hilbert_space_size(m + 2, n - 1)))
#     ))
#     smsp = block_diag((smsp, d))
#     i += 1
#
# d = block_diag((
#     diags([1] * (hilbert_space_size(m, n - 1) + hilbert_space_size(m + 1, n - 1))),
#     coo_matrix((hilbert_space_size(m - 1, n - 1) * hilbert_space_size(m, n),
#                 hilbert_space_size(m + 2, n - 1)))
# ))

# H = np.asarray([[0, 0, 0, 1],
#                 [0, 0, 1, 0],
#                 [0, 1, 0, 0],
#                 [1, 0, 0, 0]])
#
# E, V = np.linalg.eigh(H)
# print(E)
# print(V)
