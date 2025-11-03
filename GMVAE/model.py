from torch import nn
import torch
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(
        self, input_dim=784, hidden_dim=400, latent_dim=20,
        likelihood="gaussian"
    ):
        super(VAE, self).__init__()
        self.input_dim = input_dim # input dim depends on dataset
        self.hidden_dim = hidden_dim # hidden dim for first linear layer
        self.latent_dim = latent_dim # dimension in latent space

        # Encoder layers
        # first linear
        self.linear_encode = nn.Linear(input_dim, hidden_dim)
        # Mean of latent space
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        # Log variance of latent space  
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)  

        # Decoder layers
        # first linear
        self.linear_decode = nn.Linear(latent_dim, hidden_dim)
        # map to likelihood parameters
        self.linear_likelihood = nn.Linear(
            hidden_dim, input_dim
        )
        if likelihood not in ["gaussian", "bernoulli"]:
            raise ValueError("Likelihood must be either 'gaussian'"
                              "or 'bernoulli'"
            )
        else:
            self.likelihood = likelihood
            
    def encode(self, x):
            h = F.relu(self.linear_encode(x))
            return self.linear_mu(h), self.linear_logvar(h)


    def decode(self, z):
            h = F.relu(self.linear_decode(z))
            if self.likelihood == "gaussian":
                return self.linear_likelihood(h)
            elif self.likelihood == "bernoulli":
                return torch.sigmoid(self.linear_likelihood(h))
            
            
    def sample(self, mu, logvar):
        """Sample from Gaussian posterior of z with reparametrization"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x) # get posterior parameters
        z = self.sample(mu, logvar) # sample with reparametrization
        recon_x = self.decode(z) # get likelihood parameters
        return recon_x, mu, logvar
        
def loss_function(recon_x, x, mu, logvar, likelihood="gaussian", beta=1.0):
    """Computes the VAE loss."""
    
    if likelihood == "gaussian":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    elif likelihood == "bernoulli":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    else:
        raise ValueError("Likelihood must be either 'gaussian' or 'bernoulli'")
    # KL divergence between the posterior and the prior (both Gaussian)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * KLD
        
def train(epoch, model, train_loader, optimizer, device):
    """Trains the VAE for one epoch."""
    model.train()
    train_loss = 0
    likelihood = model.likelihood
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device).view(-1, model.input_dim)  # Flatten the data
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, likelihood=likelihood)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    average_loss = train_loss / len(train_loader.dataset)
    return average_loss
        
        
