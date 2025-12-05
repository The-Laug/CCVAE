import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch import Tensor
import os
import importlib
import plotting
import numpy as np
importlib.reload(plotting)
from plotting import make_new_vae_plots
import compare
importlib.reload(compare)
from compare import save_performance 
import random

from CCVAE_test_permutation import CCVAE, generate_plots


EPS = 1e-6
NUM_EPOCHS = 50
learning_rate = 5e-4
hidden_dim = 512

if __name__ == "__main__":


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Current device:", device)
    print(f"Using learning rate: {learning_rate}, hidden dim: {hidden_dim}")

    transform = T.Compose([
    T.ToTensor(), 
    lambda t: t.view(-1) 
        ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def filter_mnist(dataset, keep):
        mask = torch.isin(dataset.targets, torch.tensor(keep))
        dataset.data = dataset.data[mask]
        dataset.targets = dataset.targets[mask]
        return dataset

    numbers = 6
    nums = list(range(0, numbers)) if 'numbers' in locals() else list(range(10))
    latent_dim = len(nums)

    if latent_dim == 3:
        nums = [1,2,4]

    trainset = filter_mnist(trainset, keep=nums)
    testset = filter_mnist(testset, keep=nums)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=128, shuffle=True, num_workers = 0)

    input_dim = 28 * 28
    
    print(f"Training CCVAE with latent dim: {latent_dim}")
    print(f"used numbers: {nums}")
    print(f"epochs : {NUM_EPOCHS}")
    
    model = CCVAE(input_dim=input_dim,
                  enc_hidden_dims=hidden_dim,
                  dec_hidden_dims=hidden_dim,
                  latent_dim=latent_dim).to(device)
    
    test_name = f"hyperparam_test_lr_{learning_rate}_hd_{hidden_dim}_ld_{latent_dim}"  

    path = f"saves/CCVAE/permutation/{test_name}"
    performance_path = f"saves/CCVAE/permutation/{test_name}/performance"
    plots_path = f"saves/CCVAE/permutation/{test_name}/plots"

    if os.path.exists(f"{path}/CCVAE_model_e_{NUM_EPOCHS}_ld_{latent_dim}.pth"): 

        model.load_state_dict(torch.load(f"{path}/CCVAE_model_e_{NUM_EPOCHS}_ld_{latent_dim}.pth", weights_only=True, map_location=device))
        generate_plots(model, test_loader, device, latent_dim, single_graph='Simplex')
    else:
        print("Could not find model")