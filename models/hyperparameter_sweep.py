import itertools
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import os
import csv

# Import your modules
from CCVAE_script import CCVAE, VariationalInference, train_from_scratch
from CCVAE_script import test_loop

# ===============================
# 1. Prepare dataset
# ===============================

def load_data():
    transform = T.Compose([
        T.ToTensor(),
        lambda t: t.view(-1)
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # Filter digits 1,2,3,4,5,6 (same as your script)
    def filter_mnist(dataset, keep):
        mask = torch.isin(dataset.targets, torch.tensor(keep))
        dataset.data = dataset.data[mask]
        dataset.targets = dataset.targets[mask]
        return dataset

    trainset = filter_mnist(trainset, [1,2,3,4,5,6])
    testset = filter_mnist(testset, [1,2,3,4,5,6])

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    return train_loader, test_loader


# ===============================
# 2. Define hyperparameter search space
#   3 × 2 × 2 × 2 × 1 = 24 runs
# ===============================

latent_dims      = [6]        # 1 choice
free_bits        = [0.0, 20.0]      # 2 choices
betas            = [1.0, 5.0]       # 2 choices
warmups          = [0, 50]          # 2 choices
hidden_sizes     = [500]            # 1 choice (keep fixed to stay at ~25 tests)

sweep = list(itertools.product(
    latent_dims,
    free_bits,
    betas,
    warmups,
    hidden_sizes
))

# ~24 runs
print("Total runs:", len(sweep))


# ===============================
# 3. Run sweep
# ===============================

def run_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data()

    results_path = "sweep_results.csv"
    write_header = not os.path.exists(results_path)

    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "latent_dim", "free_bits", "beta", "warmup",
                "hidden", "train_loss", "test_loss",
                "train_kl", "test_kl"
            ])

        for (latent_dim, C_free, beta, warmup, hidden) in sweep:
            
            print("\n============================")
            print(f"Running model: LD={latent_dim}, C_free={C_free}, β={beta}, warmup={warmup}")
            print("============================\n")

            model = CCVAE(
                input_dim=28*28,
                enc_hidden_dims=hidden,
                dec_hidden_dims=hidden,
                latent_dim=latent_dim
            ).to(device)

            vi = VariationalInference(
                zero_beta_epochs=0,
                base_beta=beta,
                warmup_epochs=warmup,
                max_beta=beta,
                C_free=C_free
            ).to(device)

            # Train for fewer epochs to keep sweep fast (adjust if needed)
            global NUM_EPOCHS
            NUM_EPOCHS = 20

            trained_model = train_from_scratch(
                model, vi, train_loader, test_loader, device
            )

            # Evaluate
            test_loss, test_recon, test_kl = test_loop(trained_model, vi, test_loader, device)
            train_loss, train_recon, train_kl = test_loop(trained_model, vi, train_loader, device)

            writer.writerow([
                latent_dim, C_free, beta, warmup,
                hidden, train_loss, test_loss,
                train_kl, test_kl
            ])

    print("\nSweep finished!")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    run_sweep()
