"""
Evaluation script for the AMAG model.

Loads a trained model and evaluates it on the validation set,
printing MSE / MAE metrics and saving results to results.json.

Usage:
    python baselines/training/AMAG/evaluation.py --monkey affi
    python baselines/training/AMAG/evaluation.py --monkey beignet

Author: Joe Liao, National Central University (NCU)
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from model import AMAG
from utils import (
    get_training_args,
    NeuralForecastingDataset,
    load_data,
    evaluate_metrics,
    save_results,
    set_seed,
)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Run inference on the full dataloader and collect predictions.

    Returns:
        predictions: np.ndarray of shape (num_samples, 20, N)
        targets:     np.ndarray of shape (num_samples, 20, N)
    """
    model.eval()
    all_preds = []
    all_targets = []

    for x, target in tqdm(dataloader, desc="Evaluating"):
        x = x.to(device)
        output = model(x).cpu().numpy()
        all_preds.append(output)
        all_targets.append(target.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return preds, targets


def main():
    args = get_training_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(__file__).resolve().parent

    # ---- Load model ----------------------------------------------------
    model = AMAG(monkey_name=args.monkey).to(device).float()

    # Try loading from submissions folder first, fall back to local
    submissions_path = (
        save_dir.parent / "submissions" / "AMAG" / f"model_{args.monkey}.pth"
    )
    local_path = save_dir / f"model_{args.monkey}.pth"
    weight_path = submissions_path if submissions_path.exists() else local_path

    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from {weight_path}")
    else:
        print(f"WARNING: No weights found at {submissions_path} or {local_path}")

    # ---- Load data -----------------------------------------------------
    _, val_data = load_data(args)
    print(f"Validation samples: {len(val_data)}")

    val_dataset = NeuralForecastingDataset(val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ---- Evaluate ------------------------------------------------------
    preds, targets = evaluate(model, val_loader, device)
    metrics = evaluate_metrics(preds, targets)

    print("\n===== Evaluation Results =====")
    print(f"  MSE (full horizon) : {metrics['mse_full']:.6f}")
    print(f"  MAE (full horizon) : {metrics['mae_full']:.6f}")
    print(f"  MSE (pred only)    : {metrics['mse_pred']:.6f}")
    print(f"  MAE (pred only)    : {metrics['mae_pred']:.6f}")

    # Save results
    save_results(save_dir / "results.json", metrics)


if __name__ == "__main__":
    main()
