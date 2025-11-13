import torch
from torch import nn, Tensor
import numpy as np
from torch.distributions import Distribution, Bernoulli
import torch.nn.functional as F

from utils import LatentType
from distributions.reparameterized_diagonal_gaussian import ReparameterizedDiagonalGaussian
from distributions.reparameterized_dirichlet import ReparameterizedDirichlet
from distributions.reparameterized_continuous_categorical import ReparameterizedContinuousCategorical


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape: torch.Size, latent_features: int, latent_type: LatentType, tau_start:float = 2.0) -> None:
        super(VariationalAutoencoder, self).__init__()
        
        self.latent_type = latent_type
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        
        # temperature for Gumbel–Softmax
        self.tau = tau_start

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(), 
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(out_features=2 * latent_features if latent_type == LatentType.GAUSSIAN else latent_features)
        )

        self.fc_dec = nn.Linear(latent_features, 32 * 28 * 28)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=1, padding=2),
        )  
        
        if self.latent_type == LatentType.GAUSSIAN:
            self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2 * latent_features])))
        elif self.latent_type == LatentType.DIRICHLET:
            self.register_buffer('alpha_prior', torch.ones(torch.Size([1, latent_features])))
        else:  # continuous categorical
            self.register_buffer('logits_prior', torch.zeros(torch.Size([1, latent_features])))

    def posterior(self, x: torch.Tensor):
        h_x = self.encoder(x)
        if self.latent_type == LatentType.GAUSSIAN:
            mu, log_sigma = h_x.chunk(2, dim=-1)
            return ReparameterizedDiagonalGaussian(mu, log_sigma)
        elif self.latent_type == LatentType.DIRICHLET:
            alpha = F.softplus(h_x) + 1e-4
            return ReparameterizedDirichlet(alpha)
        else:  # continuous categorical
            return ReparameterizedContinuousCategorical(h_x)

    def prior(self, batch_size: int = 1):
        if self.latent_type == LatentType.GAUSSIAN:
            prior_params = self.prior_params.expand(batch_size, -1)
            mu, log_sigma = prior_params.chunk(2, dim=-1)
            return ReparameterizedDiagonalGaussian(mu, log_sigma)
        elif self.latent_type == LatentType.DIRICHLET:
            alpha = self.alpha_prior.expand(batch_size, -1)
            return ReparameterizedDirichlet(alpha)
        else:  # continuous categorical
            logits = self.logits_prior.expand(batch_size, -1)
            return ReparameterizedContinuousCategorical(logits)
        

    def observation_model(self, z: torch.Tensor) -> Distribution:
        h = self.fc_dec(z)                       # (B, 32*28*28)
        h = h.view(-1, 32, 28, 28)               # reshape to feature map
        px_logits = self.decoder(h)
        px_logits = px_logits.view(-1, *self.input_shape)
        return Bernoulli(logits=px_logits, validate_args=False)


    def forward(self, x: torch.Tensor):

        # encode
        qz = self.posterior(x)

        # prior
        pz = self.prior(batch_size=x.size(0))


        z = qz.rsample()

        # decode
        px = self.observation_model(z)

        return {"px": px, "pz": pz, "qz": qz, "z": z}
    
    def sample_from_prior(self, batch_size: int = 100):
        """Sample z ~ p(z) and return p(x|z)."""
        pz = self.prior(batch_size=batch_size)

        # sample latent variable
        z = pz.rsample() if hasattr(pz, "rsample") else pz.sample()

        # decode
        px = self.observation_model(z)

        return {"px": px, "pz": pz, "z": z}

    def sample_from_posterior(self, x: torch.Tensor):
        """Sample z ~ q(z|x) and return p(x|z)."""

        # encode
        qz = self.posterior(x)

        # sample latent variable
        z = qz.rsample() if hasattr(qz, "rsample") else qz.sample()

        # decode
        px = self.observation_model(z)

        return {"px": px, "qz": qz, "z": z}