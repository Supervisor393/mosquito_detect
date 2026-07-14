"""Deterministic in-distribution evaluation with source-overlap diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from dataset import MosquitoDataset, collate_fn
from evaluate_ood import compute_metrics, plot_confusion, plot_threshold_tradeoff
from model import get_model


def split_source_names(data_dir: Path, metadata: pd.DataFrame, split: str) -> set[str]:
    ids = set()
    for label in ("mosquito", "no_mosquito"):
        ids.update(path.stem for path in (data_dir / split / label).glob("*.wav"))
    id_to_name = dict(zip(metadata["id"].astype(str), metadata["name"].astype(str)))
    return {id_to_name[sample_id] for sample_id in ids if sample_id in id_to_name}


def evaluate(
    model_path: Path,
    model_type: str,
    data_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    threshold: float,
    device_name: str,
) -> None:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    # is_train=False is important: it selects the deterministic center crop.
    dataset = MosquitoDataset(data_dir / "test", is_train=False)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    model = get_model(model_type, num_classes=config.NUM_CLASSES).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    probabilities = []
    labels = []
    with torch.no_grad():
        for features, batch_labels in loader:
            outputs = model(features.to(device))
            probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.numpy())

    labels_array = np.asarray(labels, dtype=int)
    probabilities_array = np.asarray(probabilities, dtype=float)
    predictions = (probabilities_array >= threshold).astype(int)

    metadata = pd.read_csv(metadata_path)
    metadata["id_string"] = metadata["id"].astype(str)
    metadata_by_id = metadata.set_index("id_string")
    train_source_names = split_source_names(data_dir, metadata, "train")
    sample_ids = [Path(path).stem for path, _ in dataset.samples]
    source_names = [str(metadata_by_id.loc[sample_id, "name"]) for sample_id in sample_ids]
    source_seen = np.asarray([name in train_source_names for name in source_names])

    results = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "path": [path for path, _ in dataset.samples],
            "source_name": source_names,
            "source_seen_in_train": source_seen,
            "label": labels_array,
            "mosquito_probability": probabilities_array,
            "prediction": predictions,
            "correct": predictions == labels_array,
        }
    )

    metrics: dict[str, object] = {
        "model_path": str(model_path.resolve()),
        "model_type": model_type,
        "device": str(device),
        "crop": "deterministic center 3-second crop",
        "overall": compute_metrics(labels_array, probabilities_array, threshold),
    }
    for name, mask in (("source_seen_in_train", source_seen), ("source_unseen_in_train", ~source_seen)):
        metrics[name] = compute_metrics(labels_array[mask], probabilities_array[mask], threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_confusion(
        labels_array,
        probabilities_array,
        threshold,
        output_dir / "confusion_matrix.png",
        evaluation_name="In-distribution",
    )
    plot_threshold_tradeoff(
        labels_array,
        probabilities_array,
        output_dir / "threshold_tradeoff.png",
        evaluation_name="In-distribution",
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Detailed ID results saved to: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--model_type", choices=("efficient", "simple"), default="efficient")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--metadata_path",
        type=Path,
        default=Path("data/metadata/neurips_2021_zenodo_0_0_1.csv"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("models/id_evaluation"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    evaluate(
        args.model_path,
        args.model_type,
        args.data_dir,
        args.metadata_path,
        args.output_dir,
        args.threshold,
        args.device,
    )


if __name__ == "__main__":
    main()
