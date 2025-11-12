from torch import Tensor
import torch
from torch.distributions import Distribution
import math

class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(z | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, I)`.
    """
    def __init__(self, mu: Tensor, log_sigma: Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu`: {mu.shape} and `log_sigma`: {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.log_sigma = log_sigma
        self.sigma = log_sigma.exp()
        
    def sample_epsilon(self) -> Tensor:
        """ε ~ N(0, I)"""
        return torch.empty_like(self.mu).normal_()
        
    def sample(self) -> Tensor:
        """Sample z ~ N(z | mu, sigma) (without gradients)"""
        with torch.no_grad():
            return self.rsample()
        
    def rsample(self) -> Tensor:
        """Sample z ~ N(z | mu, sigma) (with the reparameterization trick)"""
        eps = self.sample_epsilon()
        return self.mu + self.sigma * eps
    
    def log_prob(self, z: Tensor) -> Tensor:
        log_p = -0.5 * (((z - self.mu) / self.sigma) ** 2 + 2 * self.log_sigma + math.log(2 * math.pi))
        if log_p.ndim > 1:
            log_p = log_p.sum(dim=-1)
        return log_p
