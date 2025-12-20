"""
evaluate_accuracy.py
----------------------------------
Evaluate action classification accuracy
for shadow puppet action recognition.

Metric: Action Classification Accuracy (%)
"""

import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -----------------------------
# Load predictions
# -----------------------------
def load_predictions(pred_path):
    """
    Load prediction results.

    Expected JSON format:
    {
        "sample_id_1": {"gt": "walk", "pred": "walk"},
        "sample_id_2": {"gt": "jump", "pred": "run"},
        ...
    }
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    y_true, y_pred = [], []
    for _, item in data.items():
        y_true.append(item["gt"])
        y_pred.append(item["pred"])

    return y_true, y_pred


# -----------------------------
# Main evaluation
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate action recognition accuracy"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to prediction JSON file"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print detailed classification report"
    )
    parser.add_argument(
        "--confusion",
        action="store_true",
        help="Print confusion matrix"
    )
    args = parser.parse_args()

    y_true, y_pred = load_predictions(args.predictions)

    acc = accuracy_score(y_true, y_pred)

    print("===================================")
    print(f"Number of samples : {len(y_true)}")
    print(f"Accuracy          : {acc * 100:.2f}%")
    print("Metric            : Action Classification Accuracy")
    print("===================================")

    if args.report:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))

    if args.confusion:
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
