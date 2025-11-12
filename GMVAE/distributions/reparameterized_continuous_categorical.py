#-------------------------OLD IMPLEMENTATION------------------------

# import torch
# import torch.nn.functional as F
# from torch.distributions import Distribution
# class ReparameterizedContinuousCategorical(Distribution):
#     """
#     A reparameterized Continuous Categorical (Gumbel-Softmax) distribution.
#     Samples from a relaxed categorical using the Gumbel-Softmax trick.
#     """
#     def __init__(self, logits: torch.Tensor, temperature: float = 1.0):
#         super().__init__()
#         self.logits = logits
#         self.temperature = temperature

#     def sample(self) -> torch.Tensor:
#         """Sample without gradients."""
#         with torch.no_grad():
#             return F.gumbel_softmax(self.logits, tau=self.temperature, hard=False, dim=-1)

#     def rsample(self) -> torch.Tensor:
#         """Reparameterized sample using the Gumbel-Softmax trick."""
#         return F.gumbel_softmax(self.logits, tau=self.temperature, hard=False, dim=-1)

#     def log_prob(self, z: torch.Tensor) -> torch.Tensor:
#         """Compute approximate log-probability under the relaxed categorical."""
#         # This is not an exact density but a differentiable approximation
#         log_q = torch.sum(z * F.log_softmax(self.logits, dim=-1), dim=-1)
#         return log_q
#----------------------------------------------------------------------------------

import torch
from torch.distributions import Distribution
import tensorflow as tf
import numpy as np
from distributions.cc.cc_torch import sample_cc_ordered_torch, cc_log_prob_torch, sample_cc_reparam_batch_torch
# from cc.cc_torch import sample_cc_ordered_torch, cc_log_prob_torch, sample_cc_reparam_batch_torch
# import sys
# import os

# Go one directory up (from GMVAE to CCVAE) and then into cc
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'cc')))

class ReparameterizedContinuousCategorical(Distribution):
    """
    Continuous Categorical (CC) distribution.
    Sampling uses the differentiable ordered rejection sampler (Algorithm 2).
    Log-probability uses the cc_log_prob function from cc_funcs.
    """

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits

    def rsample(self, n_samples=1):
        """
        Differentiable reparameterized sample using Algorithm 2 (reparameterized rejection sampler).
        """
        lam = torch.softmax(self.logits, dim=-1)
        lam_batch = lam.repeat(n_samples, 1)
        samples = sample_cc_reparam_batch_torch(lam_batch)
        return samples
    
    def sample_exact(self, n_samples=1):
        """
        Non-differentiable exact sampler (Algorithm 1).
        Used only after training for evaluation or visualization.
        """
        lam = torch.softmax(self.logits, dim=-1)
        lam_batch = lam.repeat(n_samples, 1)
        samples = sample_cc_ordered_torch(lam_batch)
        return samples.detach()
    
    def sample(self, n_samples=1):
        return self.rsample(n_samples).detach()

    def log_prob(self, z: torch.Tensor):
        lam = torch.softmax(self.logits, dim=-1)
        eta = torch.log(lam[:, :-1]) - torch.log(lam[:, -1].unsqueeze(-1))
        return cc_log_prob_torch(z, eta)
