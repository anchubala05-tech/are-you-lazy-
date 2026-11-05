"""
predict.py
----------
Loads the trained model + scaler and predicts procrastination probability
for one task, either from CLI flags or an interactive prompt.

Examples:
    # Fully specified via flags
    python src/predict.py \\
        --hours_until_deadline 36 --task_difficulty 7 --hours_slept 5.5 \\
        --cups_of_coffee 2 --open_browser_tabs 23 --social_media_checks_per_hour 6 \\
        --mood_score 4 --previous_completion_rate 0.7 --is_weekend 0 --noise_level 6

    # Interactive — just answer the prompts
    python src/predict.py --interactive
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import FEATURE_COLUMNS, Scaler
from src.model import ProcrastinationNet


FRIENDLY_PROMPTS = {
    "hours_until_deadline": "Hours until the deadline",
    "task_difficulty": "Task difficulty (1-10)",
    "hours_slept": "Hours slept last night",
    "cups_of_coffee": "Cups of coffee today",
    "open_browser_tabs": "Open browser tabs right now",
    "social_media_checks_per_hour": "Social media checks per hour",
    "mood_score": "Mood right now (1-10)",
    "previous_completion_rate": "Your on-time completion rate historically (0-1)",
    "is_weekend": "Is it the weekend? (1=yes, 0=no)",
    "noise_level": "Environment noise/distraction level (1-10)",
}


def load_model(model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device, weights_only=False)
    model = ProcrastinationNet(n_features=checkpoint["n_features"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scaler = Scaler.load(os.path.join(model_dir, "scaler.json"))
    return model, scaler, device


def predict_one(model, scaler, device, feature_dict: dict) -> float:
    x = np.array([[feature_dict[c] for c in FEATURE_COLUMNS]], dtype=np.float32)
    x = scaler.transform(x)
    x_t = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x_t)).item()
    return prob


def verdict(prob: float) -> str:
    if prob < 0.2:
        return "You'll probably just do the thing. Suspicious, but great."
    if prob < 0.4:
        return "Mild procrastination risk. You might open one extra tab. Fine."
    if prob < 0.6:
        return "Coin flip. Close the tabs now, while you still can."
    if prob < 0.8:
        return "High risk. You will reorganize your desktop icons before starting."
    return "Severe risk. Deadline is basically a suggestion to you at this point."


def main():
    parser = argparse.ArgumentParser(description="Predict procrastination probability for one task.")
    parser.add_argument("--model_dir", type=str, default="saved_models")
    parser.add_argument("--interactive", action="store_true")
    for col in FEATURE_COLUMNS:
        parser.add_argument(f"--{col}", type=float, default=None)
    args = parser.parse_args()

    model, scaler, device = load_model(args.model_dir)

    if args.interactive:
        feature_dict = {}
        print("Answer a few questions about the task at hand:\n")
        for col in FEATURE_COLUMNS:
            val = input(f"  {FRIENDLY_PROMPTS[col]}: ")
            feature_dict[col] = float(val)
    else:
        missing = [c for c in FEATURE_COLUMNS if getattr(args, c) is None]
        if missing:
            parser.error(
                f"Missing required feature(s): {', '.join(missing)}. "
                "Pass them as flags, or use --interactive."
            )
        feature_dict = {c: getattr(args, c) for c in FEATURE_COLUMNS}

    prob = predict_one(model, scaler, device, feature_dict)

    print(f"\nProcrastination probability: {prob:.1%}")
    print(verdict(prob))


if __name__ == "__main__":
    main()
