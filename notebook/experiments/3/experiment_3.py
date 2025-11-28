# -*- coding: utf-8 -*-

# Experiment 3: Simple Harmonic Motion approximation x(t) = A cos(ω t) using MLP vs KAN.


# Section 0: Imports, device selection, and metric helpers.

import os

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score

# Use a non-interactive matplotlib backend to avoid display issues.
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("Agg")

# Prefer MPS on Apple Silicon if available, otherwise fall back to CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)


def rmse(y_true, y_pred):
    """Root Mean Squared Error between two torch tensors on the same scale."""
    return torch.sqrt(nn.MSELoss()(y_pred, y_true)).item()


def r2_score_torch(y_true, y_pred):
    """Wrapper around sklearn r2_score for torch tensors."""
    return r2_score(y_true.detach().cpu().numpy(),
                    y_pred.detach().cpu().numpy())


torch.manual_seed(0)

# Section 1: Generate synthetic SHM data for x(t) = A cos(ω t).

def generate_shm_dataset(
    n_samples=6000,
    A_range=(0.5, 5.0),
    w_range=(0.5, 10.0),
    t_range=(0.0, 5.0),
    noise_std=0.0,
):
    """
    Generate samples of x(t) = A * cos(ω t) with optional Gaussian noise.
    Returns:
        A, w, t, x  (each of shape [N, 1], on CPU)
    """
    A = torch.empty(n_samples, 1).uniform_(*A_range)
    w = torch.empty(n_samples, 1).uniform_(*w_range)
    t = torch.empty(n_samples, 1).uniform_(*t_range)

    x = A * torch.cos(w * t)

    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)

    return A, w, t, x


A, w, t, x = generate_shm_dataset(
    n_samples=6000,
    A_range=(0.5, 5.0),
    w_range=(0.5, 10.0),
    t_range=(0.0, 5.0),
    noise_std=0.0,
)

idx = torch.randperm(len(x))
n_train = 5000
train_idx = idx[:n_train]
test_idx = idx[n_train:]

A_tr, w_tr, t_tr, x_tr = A[train_idx], w[train_idx], t[train_idx], x[train_idx]
A_te, w_te, t_te, x_te = A[test_idx], w[test_idx], t[test_idx], x[test_idx]

y_tr = x_tr.to(device)
y_te = x_te.to(device)

print("Train samples:", len(y_tr), "| Test samples:", len(y_te))

# Section 2: Physics-informed feature θ = ω t and normalization of inputs/targets.

theta_tr = (w_tr * t_tr).to(device)
theta_te = (w_te * t_te).to(device)

Xtr_full = torch.cat([A_tr, w_tr, t_tr, theta_tr.cpu()], dim=1).to(device)
Xte_full = torch.cat([A_te, w_te, t_te, theta_te.cpu()], dim=1).to(device)

X_mean = Xtr_full.mean(dim=0, keepdim=True)
X_std = Xtr_full.std(dim=0, keepdim=True) + 1e-8

Xtr_full_norm = (Xtr_full - X_mean) / X_std
Xte_full_norm = (Xte_full - X_mean) / X_std

y_mean = y_tr.mean()
y_std = y_tr.std() + 1e-8

y_tr_norm = (y_tr - y_mean) / y_std
y_te_norm = (y_te - y_mean) / y_std

# Define feature subsets: MLP uses [A, ω, t, θ], KAN-aux uses [A, θ], KAN-raw uses [A, ω, t].
Xtr_mlp = Xtr_full_norm
Xte_mlp = Xte_full_norm

Xtr_aux = Xtr_full_norm[:, [0, 3]]
Xte_aux = Xte_full_norm[:, [0, 3]]

Xtr_raw = Xtr_full_norm[:, :3]
Xte_raw = Xte_full_norm[:, :3]

# Section 3: Improved MLP baseline on normalized (A, ω, t, θ).

class SHMMLP(nn.Module):
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.layers(x)


mlp = SHMMLP(in_dim=4, hidden=64).to(device)
opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

EPOCHS_MLP = 300
for ep in range(1, EPOCHS_MLP + 1):
    mlp.train()
    opt_mlp.zero_grad()
    pred_norm = mlp(Xtr_mlp)
    loss = loss_fn(pred_norm, y_tr_norm)
    loss.backward()
    opt_mlp.step()
    if ep % 50 == 0:
        print(f"[MLP] epoch {ep}/{EPOCHS_MLP} | train_loss={loss.item():.6f}")

mlp.eval()
with torch.no_grad():
    y_hat_mlp_norm = mlp(Xte_mlp)
    y_hat_mlp = y_hat_mlp_norm * y_std + y_mean

mlp_rmse = rmse(y_te, y_hat_mlp)
mlp_r2 = r2_score_torch(y_te, y_hat_mlp)

print("\n=== Improved MLP (A, ω, t, θ) with normalization ===")
print(f"RMSE: {mlp_rmse:.4f}")
print(f"R²  : {mlp_r2:.4f}")

# Section 4: KAN import to handle different pykan package layouts.

import kan

try:
    KAN = kan.KAN
except AttributeError:
    try:
        KAN = kan.KAN.KAN
    except AttributeError:
        from kan.MultKAN import MultKAN as KAN

# Section 5: KAN with auxiliary θ = ω t using normalized inputs/targets.

kan_aux = KAN(
    width=[2, 64, 64, 1],
    grid=7,
    k=3,
    seed=0,
    device=device,
)

dataset_aux = {
    "train_input": Xtr_aux,
    "train_label": y_tr_norm,
    "test_input": Xte_aux,
    "test_label": y_te_norm,
}

print("\nTraining KAN with auxiliary θ = ω t (normalized inputs/labels) ...")
kan_aux.fit(
    dataset_aux,
    opt="LBFGS",
    steps=150,
    lamb=1e-6,
    lamb_l1=0.0,
    lamb_entropy=0.0,
    update_grid=False,
)

with torch.no_grad():
    y_hat_aux_norm = kan_aux(Xte_aux)
    y_hat_aux = y_hat_aux_norm * y_std + y_mean

kan_aux_rmse = rmse(y_te, y_hat_aux)
kan_aux_r2 = r2_score_torch(y_te, y_hat_aux)

print("\n=== KAN (A, θ = ω t) with normalization ===")
print(f"RMSE: {kan_aux_rmse:.4f}")
print(f"R²  : {kan_aux_r2:.4f}")

# Section 6: KAN with raw inputs (A, ω, t) using the same normalized pipeline.

kan_raw = KAN(
    width=[3, 64, 64, 1],
    grid=7,
    k=3,
    seed=0,
    device=device,
)

dataset_raw = {
    "train_input": Xtr_raw,
    "train_label": y_tr_norm,
    "test_input": Xte_raw,
    "test_label": y_te_norm,
}

print("\nTraining KAN with raw inputs (A, ω, t) only ...")
kan_raw.fit(
    dataset_raw,
    opt="LBFGS",
    steps=150,
    lamb=1e-6,
    lamb_l1=0.0,
    lamb_entropy=0.0,
    update_grid=False,
)

with torch.no_grad():
    y_hat_raw_norm = kan_raw(Xte_raw)
    y_hat_raw = y_hat_raw_norm * y_std + y_mean

kan_raw_rmse = rmse(y_te, y_hat_raw)
kan_raw_r2 = r2_score_torch(y_te, y_hat_raw)

print("\n=== KAN (raw A, ω, t) with normalization ===")
print(f"RMSE: {kan_raw_rmse:.4f}")
print(f"R²  : {kan_raw_r2:.4f}")

# Section 7: Summary table comparing MLP and KAN variants on SHM.

print("\n=== Summary: MLP vs KAN variants on SHM x(t) = A cos(ω t) ===")
print("Model                         |   RMSE   |    R²")
print("----------------------------- | -------- | -------")
print(f"Improved MLP (A, ω, t, θ)     | {mlp_rmse:8.4f} | {mlp_r2:7.4f}")
print(f"KAN (A, θ = ω t)              | {kan_aux_rmse:8.4f} | {kan_aux_r2:7.4f}")
print(f"KAN (raw A, ω, t)             | {kan_raw_rmse:8.4f} | {kan_raw_r2:7.4f}")
