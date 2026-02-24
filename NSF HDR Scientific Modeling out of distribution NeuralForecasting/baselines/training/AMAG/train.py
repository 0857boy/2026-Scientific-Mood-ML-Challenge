"""
Training script for the AMAG model.

Usage:
    python baselines/training/AMAG/train.py --monkey affi --epochs 50
    python baselines/training/AMAG/train.py --monkey beignet --epochs 50

Author: Joe Liao, National Central University (NCU)
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import AMAG
from utils import (
    get_training_args,
    NeuralForecastingDataset,
    load_data,
    evaluate_metrics,
    set_seed,
)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """Run a single training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, target in dataloader:
        x = x.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(x)          # (B, 20, N)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    """Run validation and return loss + metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    for x, target in dataloader:
        x = x.to(device)
        target = target.to(device)

        output = model(x)
        loss = loss_fn(output, target)

        total_loss += loss.item()
        num_batches += 1
        all_preds.append(output.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = evaluate_metrics(preds, targets)
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, metrics


def train(model, train_loader, val_loader, args, save_path, device):
    """Full training loop with best-model checkpointing."""
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0

    print(f"Training AMAG ({args.monkey}) for {args.epochs} epochs …")
    tbar = tqdm(range(args.epochs), desc="Epoch", position=0)

    for epoch in tbar:
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device)

        log = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_mse_pred": f"{val_metrics['mse_pred']:.4f}",
        }
        tbar.set_postfix(log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

    print(f"Done! Best epoch: {best_epoch}  |  Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to {save_path}")

    # Reload best checkpoint
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def main():
    args = get_training_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----------------------------------------------------------
    train_data, val_data = load_data(args)
    print(f"Train samples: {len(train_data)}  |  Val samples: {len(val_data)}")

    train_dataset = NeuralForecastingDataset(train_data)
    val_dataset = NeuralForecastingDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ---- Model ---------------------------------------------------------
    model = AMAG(monkey_name=args.monkey).to(device).float()
    print(f"AMAG model ({args.monkey}) — {sum(p.numel() for p in model.parameters()):,} parameters")

    # ---- Save path (default: submissions folder) -----------------------
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(__file__).resolve().parent.parent / "submissions" / "AMAG"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"model_{args.monkey}.pth"

    # ---- Train ---------------------------------------------------------
    model = train(model, train_loader, val_loader, args, save_path, device)


if __name__ == "__main__":
    main()
