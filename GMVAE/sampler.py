import numpy as np

def inv_cdf_cont_bern(u, l):
    l = np.clip(l, 1e-6, 1 - 1e-6)
    if 0.499 < l < 0.501:
        return u
    num = np.log(u * (2 * l - 1) + 1 - l) - np.log(1 - l)
    den = np.log(l) - np.log(1 - l)
    return num / den

def sample_cont_bern(lam):
    u = np.random.uniform(0, 1, size=np.shape(lam))
    return np.array([inv_cdf_cont_bern(ui, li) for ui, li in zip(u, lam)])

def sample_cc_ordered(lam):
    lam = np.array(lam)
    sort_idx = np.argsort(-lam)
    lam_sorted = lam[sort_idx]
    K = lam_sorted.size
    accepted = False

    while not accepted:
        x = np.zeros(K)
        c = 0.0
        i = 1
        while c < 1.0 and i < K:
            denom = np.clip(lam_sorted[i] + lam_sorted[0], 1e-6, None)
            ratio = np.clip(lam_sorted[i] / denom, 1e-6, 1 - 1e-6)
            xi = sample_cont_bern(np.array([ratio]))[0]
            x[i] = xi
            c += xi
            i += 1
        if c <= 1.0:
            accepted = True

    x[0] = 1.0 - np.sum(x[1:])
    x_unsorted = np.empty_like(x)
    x_unsorted[sort_idx] = x
    return x_unsorted

def sample_cc_permutation(lam, B_inv):
    lam = np.array(lam)
    eta = lam[:-1] / lam[-1]
    B_inv_sub = B_inv[:-1, :-1]
    eta_tilde = np.matmul(B_inv_sub, eta)

    while True:
        y_prime = sample_cc_ordered(np.concatenate([eta_tilde, [1.0]]))[:-1]
        perm = np.argsort(y_prime)
        y = y_prime[perm]
        u = np.random.uniform()
        if u < 1.0:
            x = np.matmul(B_inv, np.concatenate([y, [max(1 - np.sum(y), 1e-6)]]))
            x = np.maximum(x, 1e-8)
            x /= np.sum(x)
            return x
        

def create_Omega_to_S_id_mat(size, return_inverse=True):
    mat = np.ones([size, size])
    mat_inv = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            if i + j < size - 1:
                mat[i, j] = 0.0
            if j + i == size - 2:
                mat_inv[i, j] = -1.0
            if j + i == size - 1:
                mat_inv[i, j] = 1.0
    if return_inverse:
        return mat, mat_inv
    else:
        return mat
    
if __name__ == "__main__":
    lam = np.array([2.0, 1.5, 0.5,0.4])
    _, B_inv = create_Omega_to_S_id_mat(lam.size)
    x = sample_cc_permutation(lam, B_inv)
    print("Sample:", x)
    print("Sum:", np.sum(x))
