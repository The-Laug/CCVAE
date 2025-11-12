import torch

def inv_cdf_torch(u, l):
    """Inverse CDF of the continuous Bernoulli distribution."""
    near_half = (l > 0.499) & (l < 0.501)
    safe_l = l.clamp(1e-6, 1 - 1e-6)
    num = torch.log(u * (2 * safe_l - 1) + 1 - safe_l) - torch.log(1 - safe_l)
    den = torch.log(safe_l) - torch.log(1 - safe_l)
    x = num / den
    return torch.where(near_half, u, x)

# def sample_cc_reparam_torch(lam):
#     lam = lam / (lam.sum() + 1e-8)  # normalize for stability
#     K = lam.size(0)
#     lam_K = lam[-1]
#     lam_rest = lam[:-1]
#     accepted = False
#     max_attempts = 100
#     attempts = 0
#     x = None

#     while not accepted and attempts < max_attempts:
#         attempts += 1
#         u = torch.rand(K - 1, device=lam.device, dtype=lam.dtype)
#         ratio = lam_rest / (lam_rest + lam_K + 1e-8)
#         ratio = ratio.clamp(1e-6, 1 - 1e-6)
#         x = inv_cdf_torch(u, ratio)
#         if torch.isnan(x).any() or torch.isinf(x).any():
#             continue
#         if x.sum() <= 1.0:
#             accepted = True

#     if x is None:
#         # fallback to uniform
#         x = torch.ones(K - 1, device=lam.device, dtype=lam.dtype) / (K - 1)

#     x_full = torch.cat([x, 1 - x.sum().unsqueeze(0)], dim=0)
#     return x_full

def sample_cc_reparam_torch(lam):
    """
    Reparameterized rejection sampler for the Continuous Categorical (CC) distribution.
    Implements Algorithm 2 from the paper.
    lam: Tensor of shape [K], positive entries.
    Returns: Tensor of shape [K]
    """
    K = lam.size(0)
    lam_K = lam[-1]
    lam_rest = lam[:-1]

    accepted = False
    max_attempts = 100
    attempts = 0

    while not accepted and attempts < max_attempts:
        attempts += 1
        u = torch.rand(K - 1, device=lam.device, dtype=lam.dtype)
        x = inv_cdf_torch(u, lam_rest / (lam_rest + lam_K))
        if x.sum() <= 1.0:
            accepted = True
            break

    # Construct full vector in simplex
    x_full = torch.cat([x, 1 - x.sum().unsqueeze(0)], dim=0)
    return x_full


def sample_cc_reparam_batch_torch(lam_batch):
    """
    Vectorized version for multiple lambda vectors.
    lam_batch: Tensor [n, K]
    Returns: Tensor [n, K]
    """
    samples = []
    for lam in lam_batch:
        samples.append(sample_cc_reparam_torch(lam))
    return torch.stack(samples)


def sample_cc_ordered_simple_torch(lam):
    """
    Ordered rejection sampler for the Continuous Categorical (CC) distribution.
    Based on Algorithm 2 in the paper.
    lam: 1D tensor of shape [K], positive values summing approximately to 1.
    """
    l = lam.clone()
    dim = l.size(0)

    # Sort descending
    l_sort, idx = torch.sort(l, descending=True)
    inv_idx = torch.argsort(idx)

    accepted = False
    max_attempts = 1e5
    attempts = 0

    while not accepted and attempts < max_attempts:
        attempts += 1
        U = torch.rand(dim, device=l.device, dtype=l.dtype)
        sample = torch.zeros(dim, device=l.device, dtype=l.dtype)
        cum_sum = 0.0

        for j in range(1, dim):
            sample[j] = inv_cdf_torch(U[j], l_sort[j] / (l_sort[j] + l_sort[0]))
            cum_sum += sample[j]
            if cum_sum > 1:
                break

        if cum_sum < 1:
            accepted = True

    # Fill first coordinate
    sample[0] = 1 - sample.sum()
    # Reorder back to original lambda order
    sample = sample[inv_idx]
    return sample


def sample_cc_ordered_torch(lam_batch):
    """
    Vectorized sampler for multiple lambda vectors.
    lam_batch: Tensor of shape [n, K]
    Returns: Tensor of shape [n, K]
    """
    samples = []
    for lam in lam_batch:
        samples.append(sample_cc_ordered_simple_torch(lam))
    return torch.stack(samples)

def cc_log_norm_const_torch(eta: torch.Tensor):
    """Lightweight approximate version of cc_log_norm_const in PyTorch."""
    n, K = eta.shape
    aug_eta = torch.cat([eta, torch.zeros(n, 1, device=eta.device, dtype=eta.dtype)], dim=-1)
    lam = torch.softmax(aug_eta, dim=-1)
    log_norm = torch.logsumexp(aug_eta, dim=1)
    return -log_norm + torch.sum(lam * aug_eta, dim=1)

def cc_log_prob_torch(sample: torch.Tensor, eta: torch.Tensor):
    """Full PyTorch equivalent of cc_log_prob."""
    n, K = eta.shape
    aug_eta = torch.cat([eta, torch.zeros(n, 1, device=eta.device, dtype=eta.dtype)], dim=-1)
    loglik = -torch.sum(sample * torch.nn.functional.log_softmax(aug_eta, dim=1), dim=1)
    log_norm_const = cc_log_norm_const_torch(eta)
    return loglik + log_norm_const
