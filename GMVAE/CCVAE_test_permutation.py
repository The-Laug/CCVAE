# %%
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

# --- CONSOLIDATED EPSILON ---
limit = 10
EPS = 1e-6
NUM_EPOCHS = 400
learning_rate = 5e-4
hidden_dim = 500
total_iterations = 0
total_attempts = 50000

if len(sys.argv) > 1:
    learning_rate = float(sys.argv[1])
if len(sys.argv) > 2:
    hidden_dim = int(sys.argv[2])
if len(sys.argv) > 3:
    numbers = int(sys.argv[3])


def set_seed(seed):
    """
    Sets the seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed set to: {seed}")


# ======================================================================
#  HELPER FUNCTIONS
# ======================================================================

def inv_cdf_torch(u, l):
    """
    Inverse CDF of the continuous Bernoulli distribution.
    CRITICAL FIX: This version prevents NaN generation.
    """
    u = u.clamp(EPS, 1 - EPS)
    l = l.clamp(EPS, 1 - EPS)
    
    mask_near_half = (l > 0.499) & (l < 0.501)
    l_calc = torch.where(mask_near_half, torch.full_like(l, 0.45), l)
    
    term_inner = u * (2 * l_calc - 1) + 1 - l_calc
    term_inner = term_inner.clamp(min=EPS) 
    
    num = torch.log(term_inner) - torch.log(1 - l_calc)
    den = torch.log(l_calc) - torch.log(1 - l_calc)
    
    x = num / den
    return torch.where(mask_near_half, u, x)


def sample_cc_hybrid(lam, stage1_attempts=20, stage2_attempts=7500, verbose=False):
    """
    Optimized Smart Hybrid Sampler.
    Automatically switches strategy based on confidence (heuristic).
    """
    global total_iterations
    B, K = lam.shape
    device = lam.device
    dtype = lam.dtype
    
    final_x = torch.zeros_like(lam)
    
    # --- HEURISTIC ROUTING ---
    # If max probability > 2.0/K, skip Stage 1 (Permutation)
    heuristic_threshold = 4.0 / K
    is_unbalanced = lam.max(dim=1).values > heuristic_threshold
    
    # Only "Balanced" samples try Stage 1
    active_mask_stage1 = ~is_unbalanced
    global_active_mask = torch.ones(B, dtype=torch.bool, device=device)
    
    # --- STAGE 1: Permutation Sampler ---
    perm_iters = 0
    if active_mask_stage1.any():
        K_minus_1 = K - 1
        
        # Fail Fast: limit attempts to 20
        for _ in range(stage1_attempts):
            if not active_mask_stage1.any():
                break
            perm_iters += 1
            
            active_indices = torch.nonzero(active_mask_stage1).squeeze(-1)
            lam_active = lam[active_indices]
            current_B = lam_active.size(0)
            
            lam_active = lam_active.clamp(min=EPS, max=1.0)
            eta = torch.log(lam_active[:, :-1] / (lam_active[:, -1].unsqueeze(1) + EPS))
            
            eta_diff = eta[:, :-1] - eta[:, 1:]
            eta_last = eta[:, -1:]
            eta_tilde = torch.cat([eta_diff, eta_last], dim=1)
            
            u = torch.rand(current_B, K_minus_1, device=device, dtype=dtype)
            l_cb = torch.sigmoid(eta_tilde)
            y_prime = inv_cdf_torch(u, l_cb)
            
            y, indices = torch.sort(y_prime, dim=1)
            
            term_1 = torch.sum(eta_tilde * (y - y_prime), dim=1)
            
            eta_tilde_d = eta_tilde.double()
            eta_tilde_flip = eta_tilde_d.flip(dims=[1])
            suffix_sum_eta = torch.cumsum(eta_tilde_flip, dim=1).flip(dims=[1])
            zeros = torch.zeros(current_B, 1, device=device, dtype=torch.double)
            suffix_sum_eta = torch.cat([suffix_sum_eta, zeros], dim=1)
            
            eta_gathered = torch.gather(eta_tilde_d, 1, indices)
            eta_gathered_flip = eta_gathered.flip(dims=[1])
            suffix_sum_gathered = torch.cumsum(eta_gathered_flip, dim=1).flip(dims=[1])
            suffix_sum_gathered = torch.cat([suffix_sum_gathered, zeros], dim=1)
            
            diffs = suffix_sum_eta - suffix_sum_gathered
            log_kappa, _ = torch.max(diffs, dim=1)
            log_alpha = term_1 - log_kappa.to(dtype)
            
            log_u_accept = torch.log(torch.rand(current_B, device=device, dtype=dtype) + EPS)
            accepted_local = log_u_accept < log_alpha
            
            if accepted_local.any():
                y_acc = y[accepted_local]
                indices_acc = indices[accepted_local]
                
                y_padded = torch.cat([torch.zeros(y_acc.size(0), 1, device=device, dtype=dtype), y_acc], dim=1)
                x_first_km1 = y_padded[:, 1:] - y_padded[:, :-1]
                x_last = 1.0 - y_acc[:, -1:]
                x_acc = torch.cat([x_first_km1, x_last], dim=1)
                
                N_acc = indices_acc.size(0)
                pivot_index = torch.full((N_acc, 1), K_minus_1, device=device, dtype=indices_acc.dtype)
                indices_full = torch.cat([indices_acc, pivot_index], dim=1) 
                
                final_x_local = torch.zeros_like(x_acc)
                final_x_local.scatter_(1, indices_full, x_acc)
                
                accepted_global_indices = active_indices[accepted_local]
                final_x[accepted_global_indices] = final_x_local
                
                active_mask_stage1 = active_mask_stage1.clone()
                active_mask_stage1[accepted_global_indices] = False
                
                global_active_mask = global_active_mask.clone()
                global_active_mask[accepted_global_indices] = False

    # Stats: How many did Stage 1 handle?
    count_stage_1 = B - global_active_mask.sum().item()

    # --- STAGE 2: Ordered Sampler ---
    ordered_iters = 0
    if global_active_mask.any():
        stage2_indices = torch.nonzero(global_active_mask).squeeze(-1)
        lam_stage2 = lam[stage2_indices]
        B_stage2 = lam_stage2.size(0)
        
        # Sort ONCE
        lam_sorted, sort_indices = torch.sort(lam_stage2, dim=1, descending=True)
        lam_1 = lam_sorted[:, 0].unsqueeze(1)
        lam_rest = lam_sorted[:, 1:]
        cb_params = lam_rest / (lam_rest + lam_1 + EPS)
        
        x_rest_stage2 = torch.zeros(B_stage2, K-1, device=device, dtype=dtype)
        mask_local = torch.ones(B_stage2, dtype=torch.bool, device=device)
        
        for _ in range(stage2_attempts):
            if not mask_local.any():
                break
            ordered_iters += 1
            
            n_active = mask_local.sum()
            active_params = cb_params[mask_local]
            u = torch.rand(n_active, K-1, device=device, dtype=dtype)
            x_cand = inv_cdf_torch(u, active_params)
            
            sums = x_cand.sum(dim=1)
            accepted = (sums <= 1.0 + 1e-4)
            
            if accepted.any():
                active_loc = torch.nonzero(mask_local).squeeze(-1)
                accepted_loc = active_loc[accepted]
                x_rest_stage2[accepted_loc] = x_cand[accepted]
                mask_local = mask_local.clone()
                mask_local[accepted_loc] = False
                
        # Reconstruct Stage 2
        succeeded_mask = ~mask_local
        if succeeded_mask.any():
            x_rest_success = x_rest_stage2[succeeded_mask]
            x_1 = (1.0 - x_rest_success.sum(dim=1, keepdim=True)).clamp(min=EPS)
            x_sorted = torch.cat([x_1, x_rest_success], dim=1)
            
            sort_idx_success = sort_indices[succeeded_mask]
            x_final_stage2 = torch.zeros_like(x_sorted)
            x_final_stage2.scatter_(1, sort_idx_success, x_sorted)
            
            success_global = stage2_indices[succeeded_mask]
            final_x[success_global] = x_final_stage2
            
            global_active_mask = global_active_mask.clone()
            global_active_mask[success_global] = False

    # Stats logic
    if verbose and total_iterations < total_attempts:
        total_iterations += 1
        fail = global_active_mask.sum().item()
        s2_count = B - count_stage_1 - fail
        print(f"[Sampler] Stage 1: {count_stage_1/B*100:.1f}% ({perm_iters} it) | "
              f"Stage 2: {s2_count/B*100:.1f}% ({ordered_iters} it) | Fail: {fail}")

    # Fail-safe
    if global_active_mask.any():
        final_x[global_active_mask] = lam[global_active_mask]
        
    return final_x


def lambda_to_eta(lam: Tensor) -> Tensor:
    # Converts mean parameter lambda [B, K] to natural parameter eta [B, K-1].
    lam = lam.clamp(min=EPS, max=1.0) 
    last = lam[:, -1].unsqueeze(1)
    eta = torch.log(lam / (last + EPS))
    return eta[:, :-1]

def cc_log_norm_const_torch(eta: Tensor) -> Tensor:
    """
    Calculates log C(eta) using the Exact Formula (Eq 7 in the paper).
    Note: Calculation is done in double precision for stability.
    Includes gradient hook to zero out NaNs (Paper's suggested fix).
    """
    original_dtype = eta.dtype
    eta = eta.double()

    B, K_minus_1 = eta.shape
    K = K_minus_1 + 1
    device = eta.device
    
    # 1. Construct full eta (append 0 for the Kth component)
    eta_full = torch.cat([eta, torch.zeros(B, 1, device=device, dtype=eta.dtype)], dim=1)

    jitter = torch.arange(K, device=device) * 1e-5
    eta_full = eta_full + jitter.unsqueeze(0)

    # 3. Compute the denominator product: prod_{i!=k} (eta_i - eta_k) 
    eta_i = eta_full.unsqueeze(1) 
    eta_k = eta_full.unsqueeze(2)
    diffs = eta_i - eta_k
    
    eye_mask = torch.eye(K, device=device).bool().unsqueeze(0).expand(B, -1, -1)
    diffs[eye_mask] = 1.0 
    
    log_diffs_abs = diffs.abs().log()
    diffs_sign = diffs.sign()
    
    log_denom = log_diffs_abs.sum(dim=1) 
    denom_sign = diffs_sign.prod(dim=1) 
    
    log_terms_mag = eta_full - log_denom
    terms_sign = denom_sign
    
    # 4. Sum the terms: S = sum_k (T_k)
    max_log_mag, _ = log_terms_mag.max(dim=1, keepdim=True)
    sum_scaled = torch.sum(terms_sign * torch.exp(log_terms_mag - max_log_mag), dim=1)
    
    # 5. Multiply by (-1)^(K+1) 
    global_sign = (-1)**(K + 1)
    total_sum_signed = global_sign * sum_scaled
    
    log_inv_C = max_log_mag.squeeze() + torch.log(total_sum_signed.clamp(min=EPS))
    
    # Return log C = - log(C^-1)
    res = -log_inv_C.to(dtype=original_dtype)
    
    # --- STABILITY FIX: Zero out NaN gradients ---
    # [cite_start]As suggested in the paper[cite: 134], zero out error-inducing gradients.
    if res.requires_grad:
        res.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0))
        
    return res

def cc_log_prob_torch(sample: Tensor, eta: Tensor) -> Tensor:
    """
    Calculates the log-density p(z | eta) = eta^T * z + log C(eta).
    sample: [B, K], eta: [B, K-1]
    Returns: [B]
    """
    n, K_minus_1 = eta.shape
    aug_eta = torch.cat([eta, torch.zeros(n, 1, device=eta.device, dtype=eta.dtype)], dim=-1)
    
    # Exponent term: eta^T * z
    exponent = torch.sum(sample * aug_eta, dim=1) 
    
    # Log Normalizer term
    log_norm_const = cc_log_norm_const_torch(eta)
    
    return exponent + log_norm_const 
    

# ======================================================================
# 2. MODEL DEFINITIONS
# ======================================================================

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims),
            nn.Linear(hidden_dims, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class BernoulliDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dim, bias=False)
        )

    def forward(self, z):
        return self.net(z)

class CCVAE(nn.Module):
    def __init__(self, input_dim, enc_hidden_dims, dec_hidden_dims, latent_dim):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, enc_hidden_dims, latent_dim)
        self.decoder = BernoulliDecoder(latent_dim, dec_hidden_dims, input_dim)
        self.explore_epochs = 100 
        self.current_epoch = 1
        self.latent_dim = latent_dim

        if self.latent_dim >=limit:
            self.logits_bn = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        tau = np.maximum(0.1, 1.0 * np.exp(-0.002 * self.current_epoch))
        lam_logits = self.encoder(x) 
    
        
        if self.latent_dim>=limit:
            #"Using batchnorm on logits to prevent posterior collapse...
            lam_logits = self.logits_bn(lam_logits)

        lam = F.softmax(lam_logits/tau, dim=1)
        
        # UPDATED: Use the Robust Permutation Sampler (Iterative + Fallback)
        z = sample_cc_hybrid(lam) 

        recon_logits = self.decoder(z)
        return recon_logits, lam, z

# ======================================================================
# 3. VARIATIONAL INFERENCE CLASS 
# ======================================================================

class VariationalInference(nn.Module):
    """
    Computes the beta-ELBO loss for the CCVAE.
    """

    def __init__(self, zero_beta_epochs:float,  base_beta: float, warmup_epochs: int, max_beta: float):
        super().__init__()
        self.zero_beta_epochs = zero_beta_epochs
        self.base_beta = base_beta
        self.warmup_epochs = warmup_epochs
        self.max_beta = max_beta

    def get_beta(self, epoch: int) -> float:
        if epoch <= self.zero_beta_epochs:
            beta = 0.0
        elif epoch < self.zero_beta_epochs + self.warmup_epochs:
            progress = (epoch - self.zero_beta_epochs) / self.warmup_epochs
            beta = self.base_beta * progress
        else:
            beta = self.base_beta
            
        return min(self.max_beta, beta)

    def forward(self, x: Tensor, logits_dec: Tensor, lam: Tensor, z: Tensor, epoch: int):
        K = lam.size(1) 
        
        # --- 1. Reconstruction Loss ---
        bce = F.binary_cross_entropy_with_logits(logits_dec, x, reduction='none')
        recon_loss = bce.sum(dim=1).mean() 

        # --- 2. Monte Carlo KL Divergence ---
        eta_q = lambda_to_eta(lam)
        prior_lam = torch.full_like(lam, 1.0 / K)
        eta_p = lambda_to_eta(prior_lam)

        # Log densities p(z|q) and p(z|p)
        log_q = cc_log_prob_torch(z, eta_q) # [B]
        log_p = cc_log_prob_torch(z, eta_p) # [B]

        # KL = E_z [log q(z) - log p(z)]
        kl = (log_q - log_p).mean() 

        # --- 4. β-ELBO Loss ---
        beta = self.get_beta(epoch)
        loss = recon_loss + beta * kl
        
        return loss, recon_loss, kl
    

def montecarlo_nll(model, data_loader, device, K=500):
    """
    Monte Carlo estimate of the marginal log-likelihood.
    """
    model.eval()
    total_nll = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            B = x.size(0)

            lam_logits = model.encoder(x) 
            lam = F.softmax(lam_logits, dim=1)

            lam_expanded = lam.repeat(K, 1) 
            
            # UPDATED: Use Permutation sampler here too
            z = sample_cc_hybrid(lam_expanded) 

            logits_dec = model.decoder(z) 
            x_expanded = x.repeat(K, 1)
            
            recon_loss = F.binary_cross_entropy_with_logits(
                logits_dec, x_expanded, reduction='none'
            ).sum(dim=1)
            
            log_p_x_given_z = -recon_loss 

            eta_q = lambda_to_eta(lam_expanded) 
            prior_lam = torch.full_like(lam_expanded, 1.0 / latent_dim)
            eta_p = lambda_to_eta(prior_lam)    

            log_q_z = cc_log_prob_torch(z, eta_q) 
            log_p_z = cc_log_prob_torch(z, eta_p) 

            log_p_x_given_z = log_p_x_given_z.view(K, B)
            log_p_z = log_p_z.view(K, B)
            log_q_z = log_q_z.view(K, B)
            
            log_w = log_p_x_given_z + log_p_z - log_q_z   

            log_w_max, _ = torch.max(log_w, dim=0, keepdim=True) 
            
            log_p_x = log_w_max.squeeze(0) + torch.log(
                torch.exp(log_w - log_w_max).mean(dim=0) 
            )

            batch_nll = (-log_p_x).sum().item()
            total_nll += batch_nll
            n_samples += B

    mean_nll = total_nll / n_samples
    print(f"Estimated NLL (Monte Carlo, K={K}): {mean_nll:.4f}")
    return mean_nll


# ==========================================
# 4. TRAINING UTILS 
# ==========================================

def ccvae_loss(model: 'CCVAE', vi: VariationalInference, x: Tensor, epoch: int):
    logits_dec, lam, z = model(x) 
    loss, recon_loss, kl = vi(x, logits_dec, lam, z, epoch)
    return loss, recon_loss, kl, z

def train_loop(model: 'CCVAE', vi: VariationalInference, optimizer, train_loader, device):
    model.train()
    tot_loss = 0.0
    tot_recon = 0.0
    tot_kl = 0.0
    n_samples = 0

    for x, _ in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        
        loss, recon, kl, _ = ccvae_loss(model, vi, x, model.current_epoch)
        
        # Stability: Gradient Clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        batch_size = x.size(0)
        tot_loss += loss.item() * batch_size
        tot_recon += recon.item() * batch_size
        tot_kl += kl.item() * batch_size
        n_samples += batch_size

    mean_loss = tot_loss / n_samples
    mean_recon = tot_recon / n_samples
    mean_kl = tot_kl / n_samples
    
    loss_data.append(mean_loss)
    recon_data.append(mean_recon)
    kl_data.append(mean_kl)
    print(f"Negative ELBO: {mean_loss:.4f} | Recon Loss: {mean_recon:.4f} | KL: {mean_kl:.4f}")
    return mean_loss, mean_recon, mean_kl

loss_data = []
recon_data = []
kl_data = []

def test_loop(model: 'CCVAE', vi: VariationalInference, test_loader, device):
    model.eval()
    tot_loss = 0.0
    tot_recon = 0.0
    tot_kl = 0.0
    n_samples = 0
    with torch.no_grad():
        for x, label in test_loader:
            x = x.to(device)
            loss, recon, kl, z = ccvae_loss(model, vi, x, model.current_epoch)
            b = x.size(0)
            tot_loss += loss.item() * b
            tot_recon += recon.item() * b
            tot_kl += kl.item() * b
            n_samples += b

    mean_loss = tot_loss / n_samples
    mean_recon = tot_recon / n_samples
    mean_kl = tot_kl / n_samples

    print(f"Negative ELBO: {mean_loss:.4f} | Recon Loss: {mean_recon:.4f} | KL: {mean_kl:.4f}")
    return mean_loss, mean_recon, mean_kl

def train_from_scratch(model: 'CCVAE',vi:VariationalInference, train_loader, test_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    for epoch in range(0, NUM_EPOCHS+1):
        model.current_epoch = epoch
        print(f"Epoch {epoch}/{NUM_EPOCHS}:")
        print("--------")
        train_loop(model, vi, optimizer, train_loader, device)
        
        
        if epoch % 5 == 0:
            test_loop(model, vi, test_loader, device) 
            model.eval()
            with torch.no_grad():
                batch, _ = next(iter(test_loader))
                batch = batch[:16].to(device)
                logits_dec, _, _ = model(batch)
                recon_imgs = torch.sigmoid(logits_dec).view(-1, 1, 28, 28)
                comparison = torch.cat([batch.view(-1, 1, 28, 28), recon_imgs])

                os.makedirs("saves/CCVAE/recon_images", exist_ok=True)
                torchvision.utils.save_image(
                    comparison,
                    f"saves/CCVAE/recon_images/recon_test_epoch_{epoch:02d}.png",
                    nrow=16
                )
    return model



import importlib
import plotting
importlib.reload(plotting)
from plotting import *

import plots.simplex
importlib.reload(plots.simplex)
from plots.simplex import plot_mnist_simplex


def generate_plots(model: CCVAE, test_loader, device, latent_dims, single_graph=None):
    # 0. Collect model test data
    zs = []
    ys = []

    n = 0
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            n += y.shape[0]

            recon_logits, lam, z = model(x) 
            zs.append(z.detach().cpu())
            ys.append(y.detach().cpu())
            if n >= 2000:
                break
    zs = torch.cat(zs, dim=0)
    ys = torch.cat(ys, dim=0)

    # 1. Generate T-SNE plot
    if single_graph == None or single_graph == 'tSNE':
        print("Plotting tSNE graph...")
        fig, ax = plt.subplots()
        plot_latents(ax, zs, ys, latent_dim)   # <-- use the correct plot function
        plt.xticks([]) # remove tick labels
        plt.yticks([]) 
        plt.savefig(f"{plots_path}/cc_{latent_dim}_tSNE.svg", format="svg", bbox_inches='tight')
        #plt.show()

    # 2. Generate MDS plot
    if single_graph == None or single_graph == 'MDS':
        print("Plotting MDS graph...")
        fig, ax = plt.subplots()
        plot_mds(ax, zs, ys, latent_dim)
        plt.xticks([]) # remove tick labels
        plt.yticks([]) 
        plt.savefig(f"{plots_path}/cc_{latent_dim}_MDS.svg", format="svg", bbox_inches='tight')
        #plt.show()

    # 3. Plot one-hot latent value encoding
    if single_graph == None or single_graph == 'OH':
        fig, ax = plt.subplots()
        plot_latent_dim_wise_reconstruct(ax, model, latent_dim, device)
        plt.savefig(f"{plots_path}/cc_{latent_dim}_one-hot-latent-encoding.png", format="png", bbox_inches='tight')
        #plt.show()

    # 4. Plot simplex visualization
    if single_graph == None or single_graph == 'Simplex':
        latents = []
        labels = []
        for x, y in test_loader:
            x = x.to(device)
            with torch.no_grad():
                recon_logits, lam, z = model(x) 
            latents.append(z.cpu().numpy())
            labels.append(y.numpy())

        latent_matrix = np.vstack(latents)
        labels = np.hstack(labels)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.axis('off')
        plot_mnist_simplex(latent_matrix, labels, latent_dim=latent_dims, ax=ax)
        plt.tight_layout()
        # plt.savefig(f"{plots_path}/cc_{latent_dim}_simplex.svg", format="svg", bbox_inches='tight')
        plt.savefig(f"saves/CCVAE/permutation/testing_simplex_ld_6.svg", format="svg", bbox_inches='tight')
        #plt.show()


if __name__ == "__main__":

    set_seed(42)

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

    nums = list(range(0, numbers)) if 'numbers' in locals() else list(range(10))
    latent_dim = len(nums)

    if latent_dim == 3:
        nums = [1,2,4]

    trainset = filter_mnist(trainset, keep=nums)
    testset = filter_mnist(testset, keep=nums)
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=32, shuffle=True, num_workers = 0)

    input_dim = 28 * 28
    
    print(f"Training CCVAE with latent dim: {latent_dim}")
    print(f"used numbers: {nums}")
    print(f"epochs : {NUM_EPOCHS}")
    model = CCVAE(input_dim=input_dim,
                  enc_hidden_dims=hidden_dim,
                  dec_hidden_dims=hidden_dim,
                  latent_dim=latent_dim).to(device)

    vi = VariationalInference(
            zero_beta_epochs = 100,
            base_beta=1.0, 
            warmup_epochs=150,
            max_beta=1.0,
        ).to(device)
    
    test_name = f"hyperparam_test_lr_{learning_rate}_hd_{hidden_dim}_ld_{latent_dim}"  
    skip_load = False 

    loss_data = []
    recon_data = []
    kl_data = []

    path = f"saves/CCVAE/permutation/{test_name}"
    performance_path = f"saves/CCVAE/permutation/{test_name}/performance"
    plots_path = f"saves/CCVAE/permutation/{test_name}/plots"


    if not skip_load and os.path.exists(f"{path}/CCVAE_model_e_{NUM_EPOCHS}_ld_{latent_dim}.pth"): 
            model.load_state_dict(torch.load(f"{path}/CCVAE_model_e_{NUM_EPOCHS}_ld_{latent_dim}.pth", weights_only=True, map_location=device))
    else:
        if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
        if not os.path.exists(performance_path):
            os.makedirs(performance_path, exist_ok=True)
        
        if not os.path.exists(plots_path):
            os.makedirs(plots_path, exist_ok=True)


        model = train_from_scratch(model, vi, train_loader, test_loader, device)
        save_performance(f'{performance_path}/performance_e_{NUM_EPOCHS}_ld_{latent_dim}', loss_data, recon_data, kl_data)
        make_new_vae_plots(model, loss_data, recon_data, kl_data, model(next(iter(test_loader))[0].to(device))[0], f'{performance_path}/performance_e_{NUM_EPOCHS}_ld_{latent_dim}.png') 
        torch.save(model.state_dict(), f"{path}/CCVAE_model_e_{NUM_EPOCHS}_ld_{latent_dim}.pth")

    print("-------------")
    print("Test Results:")
    mean_test_loss, mean_test_recon, mean_test_kl = test_loop(model, vi, test_loader, device)
    
    generate_plots(model, test_loader, device, latent_dim)
    #montecarlo_nll(model, test_loader, device, K=500)

