import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
from model import VAE, loss_function, train
import numpy as np


import matplotlib.pyplot as plt


# Code is from:
#https://medium.com/@mikelgda/implementing-a-variational-autoencoder-in-pytorch-ddc0bb5ea1e7

batch_size = 128
epochs = 15
learning_rate = 1e-3

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])
pil_transform = transforms.ToPILImage()

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer
image_size = train_dataset.data[0].shape
image_size_flat = image_size[0] * image_size[1]

model = VAE(
    input_dim=image_size_flat,
    hidden_dim=400,
    latent_dim=20,
    likelihood="bernoulli"
).to(device)

#Prompt user for saved model
load_model = input("Do you want to load a saved model? (y/n): ")
if load_model.lower() == 'y':
    model.load_state_dict(torch.load("vae_mnist.pth"))
    print("Model loaded from vae_mnist.pth")
else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Training a new model.")
    # Train the model
    for epoch in range(1, epochs + 1):
        epoch_loss = train(epoch, model, train_loader, optimizer, device)
        if epoch_loss is not None:
            print(f"Epoch {epoch}, Average loss: {epoch_loss:.4f}")
    # Prompt the user for saving the model
    save_model = input("Do you want to save the trained model? (y/n): ")
    if save_model.lower() == 'y':
        torch.save(model.state_dict(), "vae_mnist.pth")
        print("Model saved as vae_mnist.pth")
    else:
        print("Model not saved.")





with torch.no_grad():

    sample, label = train_dataset[11] #random choice I made

    recon, mu, logvar = model(sample.view(-1, image_size_flat).to(device))


fig, (ax1, ax2) = plt.subplots(ncols=2)

ax1.imshow(sample.squeeze(), cmap="gray")
ax1.set_title("Original")

recon_sample = torch.bernoulli(recon.view(image_size).detach().cpu().squeeze())

ax2.imshow(recon.view(image_size).detach().cpu().squeeze(), cmap="gray")
ax2.set_title("Reconstructed")

choice1, choice2 = np.random.choice(range(len(train_dataset)), 2)

zero,_ = train_dataset[choice1]
one, _ = train_dataset[choice2]

with torch.no_grad():

    recon_zero, mu_zero, logvar_zero = model(zero.view(-1, image_size_flat).to(device))
    recon_one, mu_one, logvar_one = model(one.view(-1, image_size_flat).to(device))
    
def interpolate(mu1, mu2, n=10):
    """Interpolates between two points in latent space."""
    diff = mu2 - mu1
    step = diff / n
    samples = []
    for i in range(n):
        sample = mu1 + i * step
        samples.append(sample)
    return torch.stack(samples)

interpolated_mu = interpolate(mu_zero.detach(), mu_one.detach(), n=10)
interpolated_logvar = interpolate(logvar_zero.detach(), logvar_one.detach(), n=10)


# sampled_latent = model.reparameterize(interpolated_mu, interpolated_logvar)
# sample_decoded = model.decode(sampled_latent)

fig, (ax1, ax2) = plt.subplots(ncols=2)

ax1.imshow(zero.squeeze(), cmap="gray")
ax2.imshow(one.squeeze(), cmap="gray")


fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 5))
for img, ax in zip(sample_decoded, axes.ravel()):
    ax.imshow(img.view(image_size).detach().cpu().squeeze(), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    
    
    
    
    
    
    
    
    
    
    
plt.show()