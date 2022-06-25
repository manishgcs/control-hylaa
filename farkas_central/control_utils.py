from control import *


def extend_a_b_cont(A, B):
    A = A.copy().tolist()
    n_dim = len(A)
    A_ext = []
    for idx in range(n_dim):
        A_row = A[idx]
        A_row.append(0.0)
        A_ext.append(A_row)
    A_ext.append(np.zeros(n_dim+1, dtype=float))
    A_ext = np.array(A_ext)
    B_ext = B.copy().tolist()
    B_ext.append(np.zeros(len(B_ext[0]), dtype=float))
    B_ext = np.array(B_ext)
    return A_ext, B_ext


def extend_a_b_disc(A, B):
    A = A.copy().tolist()
    n_dim = len(A)
    A_ext = []
    for idx in range(n_dim):
        A_row = A[idx]
        A_row.append(0.0)
        A_ext.append(A_row)
    A_ext.append(np.zeros(n_dim+1, dtype=float))
    A_ext[n_dim][n_dim] = 1.0
    A_ext = np.array(A_ext)
    B_ext = B.copy().tolist()
    B_ext.append(np.zeros(len(B_ext[0]), dtype=float))
    B_ext = np.array(B_ext)
    return A_ext, B_ext


def extend_a_b(A, B, disc_dyn=False):

    if disc_dyn is False:
        return extend_a_b_cont(A, B)
    else:
        return extend_a_b_disc(A, B)


def get_input(A, B, Q, R, disc_dyn=False):
    if disc_dyn is False:
        (X1, L, G) = care(A, B, Q, R)
    else:
        (X1, L, G) = dare(A, B, Q, R)
    # print(X1, L, G)
    G = np.array(G)
    k_matrix = []
    for idx in range(len(G)):
        k_token = G[idx].copy().tolist()
        # k_token.append(0.0)
        k_matrix.append(k_token)
    k_matrix = np.array(k_matrix, dtype=float)
    return k_matrix