import itertools
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import csv
import os

from CCVAE_script import CCVAE, VariationalInference, train_loop, test_loop  


# ===================================================
# 1. DATASET
# ===================================================

def load_data():
    transform = T.Compose([
        T.ToTensor(),
        lambda t: t.view(-1)
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    keep = [1,2,3,4,5,6]

    def filt(ds):
        mask = torch.isin(ds.targets, torch.tensor(keep))
        ds.data = ds.data[mask]
        ds.targets = ds.targets[mask]
        return ds

    trainset = filt(trainset)
    testset = filt(testset)

    return (
        DataLoader(trainset, batch_size=128, shuffle=True),
        DataLoader(testset, batch_size=128, shuffle=False)
    )


# ===================================================
# 2. Sweep Space (≈25 runs)
# ===================================================

learning_rates = [1e-4, 3e-4, 1e-3]
hidden_sizes   = [256, 512]
weight_decays  = [0.0, 1e-4]
batch_norms    = [True, False]

sweep = list(itertools.product(
    learning_rates,
    hidden_sizes,
    weight_decays,
    batch_norms
))

print("Total runs:", len(sweep))  # should be 24


# ===================================================
# 3. Modified model builder to toggle BatchNorm
# ===================================================

def build_model(input_dim, hidden_dim, latent_dim, use_bn):
    class MLPEncoder(torch.nn.Module):
        def __init__(self, inp, hid, lat, bn):
            super().__init__()
            layers = [
                torch.nn.Linear(inp, hid),
                (torch.nn.BatchNorm1d(hid) if bn else torch.nn.Identity()),
                torch.nn.ReLU(),
                torch.nn.Linear(hid, lat)
            ]
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x): return self.net(x)

    class Decoder(torch.nn.Module):
        def __init__(self, lat, hid, out, bn):
            super().__init__()
            layers = [
                torch.nn.Linear(lat, hid, bias=True),
                (torch.nn.BatchNorm1d(hid) if bn else torch.nn.Identity()),
                torch.nn.ReLU(),
                torch.nn.Linear(hid, out, bias=True)
            ]
            self.net = torch.nn.Sequential(*layers)

        def forward(self, z): return self.net(z)

    # Reuse your CCVAE structure but replace the MLPs
    model = CCVAE(
        input_dim=input_dim,
        enc_hidden_dims=hidden_dim,
        dec_hidden_dims=hidden_dim,
        latent_dim=6  # fixed latent size
    )

    model.encoder = MLPEncoder(input_dim, hidden_dim, model.latent_dim, use_bn)
    model.decoder = Decoder(model.latent_dim, hidden_dim, input_dim, use_bn)

    return model


# ===================================================
# 4. Run sweep
# ===================================================

def run_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data()
    input_dim = 28*28

    results_path = "nn_hyper_sweep.csv"
    write_header = not os.path.exists(results_path)

    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "lr", "hidden", "weight_decay", "batch_norm",
                "train_elbo", "train_kl", "test_elbo", "test_kl"
            ])

        for lr, hdim, wd, bn in sweep:
            print(f"\n--- Running lr={lr}, hidden={hdim}, wd={wd}, bn={bn} ---")

            # -------------------------
            # Build model + VI module
            # -------------------------
            model = build_model(input_dim, hdim, latent_dim=6, use_bn=bn).to(device)

            vi = VariationalInference(
                zero_beta_epochs=0,
                base_beta=1.0,
                warmup_epochs=0,
                max_beta=1.0,
                C_free=0.0
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=wd
            )

            # -------------------------
            # Training loop
            # -------------------------
            NUM_EPOCHS = 15  # keep fast
            for epoch in range(NUM_EPOCHS):
                model.current_epoch = epoch
                train_loop(model, vi, optimizer, train_loader, device)

            # -------------------------
            # Evaluation
            # -------------------------
            test_loss, _, test_kl = test_loop(model, vi, test_loader, device)
            train_loss, _, train_kl = test_loop(model, vi, train_loader, device)

            writer.writerow([lr, hdim, wd, bn, train_loss, train_kl, test_loss, test_kl])

    print("\nSweep complete! Results in nn_hyper_sweep.csv")


if __name__ == "__main__":
    run_sweep()
