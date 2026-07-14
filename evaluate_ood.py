"""Evaluate a trained mosquito detector on the reproducible OOD test set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from inference import AudioProcessor, predict_audio_segment
from model import get_model


def predict_file(
    model: torch.nn.Module,
    processor: AudioProcessor,
    path: Path,
    device: torch.device,
) -> tuple[float, int]:
    waveform = processor.load_audio(path)
    segments = processor.split_into_segments(waveform)
    mosquito_probabilities = []
    for segment in segments:
        _, probabilities = predict_audio_segment(model, processor, segment, device)
        mosquito_probabilities.append(float(probabilities[1]))
    return float(np.mean(mosquito_probabilities)), len(segments)


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_mosquito": float(precision_score(labels, predictions, zero_division=0)),
        "recall_mosquito_sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "f1_mosquito": float(f1_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "false_positive_rate": float(fp / (tn + fp)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def category_metrics(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["prediction"] = (frame["mosquito_probability"] >= threshold).astype(int)
    rows = []
    for (dataset, source_class, label), group in frame.groupby(
        ["source_dataset", "source_class", "label"],
        sort=True,
    ):
        positive_rate = float(group["prediction"].mean())
        rows.append(
            {
                "source_dataset": dataset,
                "source_class": source_class,
                "true_label": int(label),
                "samples": len(group),
                "mean_mosquito_probability": float(group["mosquito_probability"].mean()),
                "recall" if label == 1 else "false_positive_rate": positive_rate,
            }
        )
    return pd.DataFrame(rows)


def plot_confusion(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    path: Path,
    evaluation_name: str = "OOD",
) -> None:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5.5, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        title=f"{evaluation_name} confusion matrix (threshold={threshold:.2f})",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No mosquito", "Mosquito"],
        yticklabels=["No mosquito", "Mosquito"],
    )
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_threshold_tradeoff(
    labels: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
    evaluation_name: str = "OOD",
) -> None:
    thresholds = np.linspace(0, 1, 201)
    sensitivities = []
    false_positive_rates = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        sensitivities.append(tp / (tp + fn) if tp + fn else 0.0)
        false_positive_rates.append(fp / (tn + fp) if tn + fp else 0.0)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(thresholds, sensitivities, label="Mosquito sensitivity (recall)", linewidth=2)
    axis.plot(thresholds, false_positive_rates, label="Background false-positive rate", linewidth=2)
    axis.set(
        xlabel="Mosquito probability threshold",
        ylabel="Rate",
        xlim=(0, 1),
        ylim=(-0.02, 1.02),
        title=f"{evaluation_name} operating-point trade-off",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate(
    model_path: Path,
    model_type: str,
    data_dir: Path,
    output_dir: Path,
    threshold: float,
    device_name: str,
) -> None:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    manifest = pd.read_csv(data_dir / "manifest.csv")
    model = get_model(model_type, num_classes=config.NUM_CLASSES).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    processor = AudioProcessor()

    probabilities = []
    segment_counts = []
    for row in manifest.itertuples(index=False):
        probability, segments = predict_file(
            model,
            processor,
            data_dir / row.relative_path,
            device,
        )
        probabilities.append(probability)
        segment_counts.append(segments)

    results = manifest.copy()
    results["mosquito_probability"] = probabilities
    results["segments"] = segment_counts
    results["prediction"] = (results["mosquito_probability"] >= threshold).astype(int)
    results["correct"] = results["prediction"] == results["label"]

    labels = results["label"].to_numpy(dtype=int)
    probability_array = results["mosquito_probability"].to_numpy(dtype=float)
    metrics = compute_metrics(labels, probability_array, threshold)
    metrics["model_path"] = str(model_path.resolve())
    metrics["model_type"] = model_type
    metrics["device"] = str(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "predictions.csv", index=False)
    category_metrics(results, threshold).to_csv(output_dir / "metrics_by_category.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_confusion(labels, probability_array, threshold, output_dir / "confusion_matrix.png")
    plot_threshold_tradeoff(labels, probability_array, output_dir / "threshold_tradeoff.png")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Detailed OOD results saved to: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--model_type", choices=("efficient", "simple"), default="efficient")
    parser.add_argument("--data_dir", type=Path, default=Path("data/ood"))
    parser.add_argument("--output_dir", type=Path, default=Path("models/ood_evaluation"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    evaluate(
        args.model_path,
        args.model_type,
        args.data_dir,
        args.output_dir,
        args.threshold,
        args.device,
    )


if __name__ == "__main__":
    main()
