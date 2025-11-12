from torch import Tensor
import torch
from torch.distributions import Distribution, Dirichlet as TorchDirichlet

class ReparameterizedDirichlet(Distribution):
    """
    A reparameterized Dirichlet distribution `Dir(z | alpha)` compatible with the
    reparameterization trick (via PyTorch's rsample on Gamma variables).
    """
    def __init__(self, alpha: Tensor):
        # alpha can be [batch_size, latent_dim] or [latent_dim]
        assert (alpha > 0).all(), "All concentration parameters (alpha) must be positive"
        self.alpha = alpha
        self._dist = TorchDirichlet(alpha)

    def sample(self) -> Tensor:
        """Sample z ~ Dir(alpha) (without gradients)."""
        with torch.no_grad():
            return self._dist.sample()

    def rsample(self) -> Tensor:
        """Sample z ~ Dir(alpha) (with gradients via reparameterization)."""
        return self._dist.rsample()

    def log_prob(self, z: Tensor) -> Tensor:
        """Compute log p(z | alpha)."""
        return self._dist.log_prob(z)
    
