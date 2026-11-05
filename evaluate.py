"""
evaluate.py
-----------
Loads the best checkpoint, runs it on the held-out test split, and reports
accuracy / precision / recall / F1 / ROC-AUC, plus a confusion matrix and
(if matplotlib is available) a training-curve + ROC plot saved to disk.

Usage:
    python src/evaluate.py --data data/procrastination_data.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_and_split, FEATURE_COLUMNS
from src.model import ProcrastinationNet


def main():
    parser = argparse.ArgumentParser(description="Evaluate the procrastination predictor on the test split.")
    parser.add_argument("--data", type=str, default="data/procrastination_data.csv")
    parser.add_argument("--model_dir", type=str, default="saved_models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NOTE: uses the same seed as train.py so the test split lines up exactly.
    _, _, test_ds, _ = load_and_split(args.data, seed=args.seed)
    test_loader = DataLoader(test_ds, batch_size=256)

    ckpt_path = os.path.join(args.model_dir, "best_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = ProcrastinationNet(n_features=checkpoint["n_features"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            probs = torch.sigmoid(model(X)).cpu().numpy().ravel()
            all_probs.append(probs)
            all_labels.append(y.numpy().ravel())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs > 0.5).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    auc = _roc_auc(labels, probs)

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")
    print(f"\nTest set size: {len(labels)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nConfusion matrix:")
    print(f"                 pred:no-procrastinate   pred:procrastinate")
    print(f"  actual:no-proc        {tn:6d}                {fp:6d}")
    print(f"  actual:proc           {fn:6d}                {tp:6d}")

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
    with open(os.path.join(args.model_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {args.model_dir}/test_results.json")

    _maybe_plot(args.model_dir, labels, probs)


def _roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), zero extra dependencies."""
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(probs) + 1)
    sum_ranks_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _maybe_plot(model_dir: str, labels: np.ndarray, probs: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots (pip install matplotlib to enable).")
        return

    history_path = os.path.join(model_dir, "history.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label="train_loss")
        axes[0].plot(epochs, history["val_loss"], label="val_loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("BCE loss")
        axes[0].set_title("Training curve")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No history.json found", ha="center")

    # Simple ROC curve from scratch (no sklearn dependency).
    thresholds = np.linspace(0, 1, 200)
    tpr, fpr = [], []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        tpr.append(tp / (tp + fn) if (tp + fn) else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) else 0)
    axes[1].plot(fpr, tpr, label="ROC curve")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="chance")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC curve")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(model_dir, "evaluation_plots.png")
    plt.savefig(out_path, dpi=130)
    print(f"Saved plots to {out_path}")


if __name__ == "__main__":
    main()
