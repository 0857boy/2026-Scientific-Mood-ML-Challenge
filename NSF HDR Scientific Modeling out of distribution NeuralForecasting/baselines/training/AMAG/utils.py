"""
Utility functions for AMAG training and evaluation.

Author: Joe Liao, National Central University (NCU)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_training_args():
    """Parse command-line arguments for training / evaluation."""
    parser = argparse.ArgumentParser(description="AMAG Training / Evaluation")
    parser.add_argument(
        "--monkey", type=str, default="affi",
        choices=["affi", "beignet"],
        help="Which monkey to train on (affi: 239 nodes, beignet: 87 nodes).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to local data directory. "
             "If not provided, data is loaded from Hugging Face.",
    )
    parser.add_argument(
        "--hf_dataset", type=str,
        default="imageomics/neural-forecasting",
        help="Hugging Face dataset identifier.",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="Hugging Face API token (if the dataset is gated).",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory to save model weights. "
             "Defaults to baselines/submissions/AMAG/.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NeuralForecastingDataset(Dataset):
    """
    Wraps pre-loaded numpy arrays for the neural forecasting task.

    Each sample is a window of shape (20, N, F):
        - First 10 time steps  = observation
        - Last  10 time steps  = ground-truth future

    The target is Feature-0 over the full 20 steps: (20, N).
    """
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: numpy array of shape (num_samples, 20, N, F)
        """
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]             # (20, N, F)
        target = x[..., 0]            # (20, N)   — Feature 0
        return x, target


def load_data(args):
    """
    Load training and validation data.

    Returns:
        train_data: np.ndarray of shape (num_train, 20, N, F)
        val_data:   np.ndarray of shape (num_val, 20, N, F)
    """
    if args.data_dir is not None:
        # ---- Local data loading ----
        data_dir = Path(args.data_dir)
        train_data = np.load(data_dir / f"{args.monkey}_train.npy")
        val_data = np.load(data_dir / f"{args.monkey}_val.npy")
    else:
        # ---- Hugging Face loading ----
        from datasets import load_dataset

        ds = load_dataset(
            args.hf_dataset,
            token=args.hf_token,
        )
        # Adapt this section to match the actual HF dataset schema
        train_data = np.array(ds["train"]["data"])
        val_data = np.array(ds["validation"]["data"])

    return train_data, val_data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return float(np.mean((predictions - targets) ** 2))


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(predictions - targets)))


def evaluate_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        predictions: (num_samples, 20, N)
        targets:     (num_samples, 20, N)

    Returns:
        dict with 'mse', 'mae', 'mse_pred_only', 'mae_pred_only'
    """
    # Full horizon (obs + pred)
    mse_full = compute_mse(predictions, targets)
    mae_full = compute_mae(predictions, targets)

    # Prediction-only horizon (last 10 steps)
    mse_pred = compute_mse(predictions[:, 10:, :], targets[:, 10:, :])
    mae_pred = compute_mae(predictions[:, 10:, :], targets[:, 10:, :])

    return {
        "mse_full": mse_full,
        "mae_full": mae_full,
        "mse_pred": mse_pred,
        "mae_pred": mae_pred,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_results(save_path: Path, metrics: dict):
    """Save evaluation metrics to a JSON file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {save_path}")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
