import torch

def inv_cdf_torch(u, l):
    """Inverse CDF of the continuous Bernoulli distribution."""
    near_half = (l > 0.499) & (l < 0.501)
    safe_l = l.clamp(1e-6, 1 - 1e-6)
    num = torch.log(u * (2 * safe_l - 1) + 1 - safe_l) - torch.log(1 - safe_l)
    den = torch.log(safe_l) - torch.log(1 - safe_l)
    x = num / den
    return torch.where(near_half, u, x)


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
