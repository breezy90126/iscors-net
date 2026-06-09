"""
Phase 1 — Pretraining with prior knowledge of cell diffusion parameters.

Teaches PISSLTauEncoder the autocorrelation pattern → (gamma, alpha) mapping
using synthetic data with PERFECT labels.  The model then needs far fewer
sparse supervision points during internal learning (Phase 2).

Set GAMMA_RANGE and ALPHA_RANGE based on your prior knowledge of the cell type
(e.g., from literature, FCS measurements, or known diffusion coefficients).
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt

from datasets.synthetic_pretrain_dataset import SyntheticPretrainDataset
from models.pissl_tau_encoder import PISSLTauEncoder
from loss.physics_loss import PhysicsInformedLoss

# ── Prior knowledge: set these from your cell biology knowledge ──────────────
GAMMA_RANGE = (0.01, 1.0)    # expected diffusion rate range
ALPHA_RANGE = (0.3, 1.8)     # expected anomalous exponent range
# ────────────────────────────────────────────────────────────────────────────

N_SAMPLES   = 10000
EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 1e-3            # higher LR than internal learning (large batch, dense GT)
NUM_TAU_CH  = 8
CHECKPOINT_DIR = "./checkpoint"
RESULT_DIR     = "./result"
PRETRAIN_PATH  = os.path.join(CHECKPOINT_DIR, "pissl_pretrained.pth")


def pretrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pretrain] device={device}  gamma={GAMMA_RANGE}  alpha={ALPHA_RANGE}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    dataset = SyntheticPretrainDataset(
        n_samples=N_SAMPLES,
        gamma_range=GAMMA_RANGE,
        alpha_range=ALPHA_RANGE,
        num_tau_channels=NUM_TAU_CH,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model     = PISSLTauEncoder(num_tau_channels=NUM_TAU_CH).to(device)
    criterion = PhysicsInformedLoss(lambda_gamma=1.0, lambda_alpha=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    history = {"loss": [], "gamma": [], "alpha": []}

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = epoch_gamma = epoch_alpha = 0.0
        pbar = tqdm.tqdm(loader, desc=f"Pretrain {epoch+1}/{EPOCHS}")

        for inputs, targets, mask in pbar:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            mask    = mask.to(device)

            optimizer.zero_grad()
            preds = model(inputs)
            loss, lg, la = criterion(preds, targets, mask)
            loss.backward()
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_gamma += lg.item()
            epoch_alpha += la.item()
            pbar.set_postfix({"L": f"{loss.item():.4f}",
                              "G": f"{lg.item():.4f}",
                              "A": f"{la.item():.4f}"})

        n = len(loader)
        history["loss"].append(epoch_loss / n)
        history["gamma"].append(epoch_gamma / n)
        history["alpha"].append(epoch_alpha / n)
        scheduler.step()
        print(f"Pretrain [{epoch+1}/{EPOCHS}] "
              f"Loss={epoch_loss/n:.4f}  G={epoch_gamma/n:.4f}  A={epoch_alpha/n:.4f}  "
              f"LR={scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), PRETRAIN_PATH)
    print(f"Pretrained model saved → {PRETRAIN_PATH}")

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"],  label="Total")
    ax.plot(history["gamma"], label="Gamma")
    ax.plot(history["alpha"], label="Alpha")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Pretrain Loss  gamma={GAMMA_RANGE}  alpha={ALPHA_RANGE}")
    ax.legend()
    fig.savefig(os.path.join(RESULT_DIR, "pretrain_loss.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Pretrain loss curve saved → {RESULT_DIR}/pretrain_loss.png")


if __name__ == "__main__":
    pretrain()
