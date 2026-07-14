"""Build a small, reproducible out-of-distribution audio test set.

Positive samples come from MosquitoSound/Wingbeats, which was captured with an
infrared optical sensor rather than the microphones used by HumBugDB. Negative
samples come from selected ESC-50 categories that are plausible false-alarm
sources for a phone mosquito detector.

The MosquitoSound feature array is more than 8 GB, but it is an uncompressed,
row-major NumPy array. This script uses HTTP range requests to download only the
selected rows (30 recordings by default).
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import soundfile as sf


MOSQUITO_X_URL = (
    "https://huggingface.co/datasets/monster-monash/MosquitoSound/resolve/main/"
    "MosquitoSound_X.npy"
)
MOSQUITO_Y_URL = (
    "https://huggingface.co/datasets/monster-monash/MosquitoSound/resolve/main/"
    "MosquitoSound_y.npy"
)
MOSQUITO_CLASSMAP_URL = (
    "https://huggingface.co/datasets/monster-monash/MosquitoSound/resolve/main/"
    "MosquitoSound_classmap.txt"
)
MOSQUITO_DATASET_URL = "https://huggingface.co/datasets/monster-monash/MosquitoSound"
MOSQUITO_SAMPLE_RATE = 6000

ESC50_META_URL = (
    "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
)
ESC50_AUDIO_BASE = (
    "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"
)
ESC50_DATASET_URL = "https://github.com/karolpiczak/ESC-50"
ESC50_CATEGORIES = (
    "insects",
    "crickets",
    "wind",
    "engine",
    "car_horn",
    "train",
    "laughing",
    "coughing",
    "church_bells",
    "airplane",
)


def request_bytes(
    url: str,
    *,
    byte_range: tuple[int, int] | None = None,
    retries: int = 4,
    timeout: int = 90,
) -> bytes:
    headers = {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                stream=byte_range is not None,
            ) as response:
                response.raise_for_status()
                if byte_range is not None and response.status_code != 206:
                    raise RuntimeError(
                        "Server did not honor the byte-range request; refusing to "
                        "download the full MosquitoSound array."
                    )
                content = response.content
                if byte_range is not None:
                    expected = byte_range[1] - byte_range[0] + 1
                    if len(content) != expected:
                        raise RuntimeError(
                            f"Expected {expected} range bytes, received {len(content)}"
                        )
                return content
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error


def download_small_file(url: str, path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(request_bytes(url))
    return path


def parse_npy_header(header_bytes: bytes) -> tuple[np.dtype, tuple[int, ...], int]:
    if header_bytes[:6] != b"\x93NUMPY":
        raise ValueError("Remote file is not a NumPy .npy array")
    major = header_bytes[6]
    if major == 1:
        header_length = struct.unpack("<H", header_bytes[8:10])[0]
        header_start = 10
    elif major in (2, 3):
        header_length = struct.unpack("<I", header_bytes[8:12])[0]
        header_start = 12
    else:
        raise ValueError(f"Unsupported NumPy file version: {major}")

    header_end = header_start + header_length
    header = ast.literal_eval(header_bytes[header_start:header_end].decode("latin1"))
    if header["fortran_order"]:
        raise ValueError("Fortran-ordered arrays are not supported")
    return np.dtype(header["descr"]), tuple(header["shape"]), header_end


def select_mosquito_rows(
    labels: np.ndarray,
    samples_per_species: int,
    seed: int,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    selected: list[tuple[int, int]] = []
    for class_label in np.unique(labels):
        candidates = np.flatnonzero(labels == class_label)
        indices = np.sort(rng.choice(candidates, size=samples_per_species, replace=False))
        selected.extend((int(index), int(class_label)) for index in indices)
    return selected


def write_mosquito_sample(
    output_dir: Path,
    index: int,
    class_label: int,
    class_name: str,
    dtype: np.dtype,
    row_shape: tuple[int, ...],
    data_offset: int,
) -> dict[str, object]:
    safe_class_name = class_name.lower()
    filename = f"mosquitosound_{safe_class_name}_{index}.wav"
    output_path = output_dir / "mosquito" / filename
    row_values = int(np.prod(row_shape))
    row_bytes = row_values * dtype.itemsize
    start = data_offset + index * row_bytes
    end = start + row_bytes - 1

    if not output_path.exists() or output_path.stat().st_size == 0:
        payload = request_bytes(MOSQUITO_X_URL, byte_range=(start, end))
        waveform = np.frombuffer(payload, dtype=dtype).reshape(row_shape).squeeze()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, waveform.astype(np.float32), MOSQUITO_SAMPLE_RATE, subtype="PCM_16")

    return {
        "relative_path": output_path.relative_to(output_dir).as_posix(),
        "label": 1,
        "label_name": "mosquito",
        "source_dataset": "MosquitoSound (Wingbeats)",
        "source_id": index,
        "source_class": class_name,
        "source_url": MOSQUITO_DATASET_URL,
        "sample_rate": MOSQUITO_SAMPLE_RATE,
        "license": "Public Domain",
        "ood_reason": "infrared optical sensor; 6 kHz; different dataset",
    }


def select_esc50_rows(metadata: pd.DataFrame) -> pd.DataFrame:
    selected = metadata[metadata["category"].isin(ESC50_CATEGORIES)].copy()
    # One deterministic clip per official fold and category gives source diversity.
    selected = (
        selected.sort_values(["category", "fold", "filename"])
        .groupby(["category", "fold"], as_index=False)
        .first()
    )
    expected = len(ESC50_CATEGORIES) * 5
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} ESC-50 clips, found {len(selected)}")
    return selected


def write_esc50_sample(output_dir: Path, row: dict[str, object]) -> dict[str, object]:
    original_filename = str(row["filename"])
    category = str(row["category"])
    filename = f"esc50_{category}_{original_filename}"
    output_path = output_dir / "no_mosquito" / filename
    audio_url = ESC50_AUDIO_BASE + quote(original_filename)

    if not output_path.exists() or output_path.stat().st_size == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(request_bytes(audio_url))

    return {
        "relative_path": output_path.relative_to(output_dir).as_posix(),
        "label": 0,
        "label_name": "no_mosquito",
        "source_dataset": "ESC-50",
        "source_id": original_filename,
        "source_class": category,
        "source_url": audio_url,
        "sample_rate": 44100,
        "license": "CC BY-NC",
        "ood_reason": "external environmental recording; realistic false-alarm probe",
    }


def build_dataset(
    output_dir: Path,
    samples_per_species: int,
    seed: int,
    workers: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "source_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    labels_path = download_small_file(MOSQUITO_Y_URL, metadata_dir / "MosquitoSound_y.npy")
    classmap_path = download_small_file(
        MOSQUITO_CLASSMAP_URL,
        metadata_dir / "MosquitoSound_classmap.txt",
    )
    esc50_meta_path = download_small_file(ESC50_META_URL, metadata_dir / "esc50.csv")

    labels = np.load(labels_path)
    classmap_frame = pd.read_csv(classmap_path)
    classmap = dict(zip(classmap_frame["class_label"], classmap_frame["class_name"]))
    selected_mosquito = select_mosquito_rows(labels, samples_per_species, seed)

    header_bytes = request_bytes(MOSQUITO_X_URL, byte_range=(0, 255))
    dtype, shape, data_offset = parse_npy_header(header_bytes)
    if shape[0] != len(labels):
        raise RuntimeError("MosquitoSound feature and label counts differ")
    row_shape = shape[1:]

    esc50_metadata = pd.read_csv(esc50_meta_path)
    selected_esc50 = select_esc50_rows(esc50_metadata)

    manifest_rows: list[dict[str, object]] = []
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, class_label in selected_mosquito:
            futures.append(
                executor.submit(
                    write_mosquito_sample,
                    output_dir,
                    index,
                    class_label,
                    classmap[class_label],
                    dtype,
                    row_shape,
                    data_offset,
                )
            )
        for row in selected_esc50.to_dict(orient="records"):
            futures.append(executor.submit(write_esc50_sample, output_dir, row))

        for future in as_completed(futures):
            manifest_rows.append(future.result())

    manifest_rows.sort(key=lambda row: (int(row["label"]), str(row["relative_path"])))
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    positives = sum(int(row["label"]) == 1 for row in manifest_rows)
    negatives = len(manifest_rows) - positives
    print(f"OOD dataset ready: {output_dir.resolve()}")
    print(f"Mosquito: {positives}; no mosquito: {negatives}")
    print(f"Manifest: {manifest_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("data/ood"))
    parser.add_argument("--samples_per_species", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    build_dataset(args.output_dir, args.samples_per_species, args.seed, args.workers)


if __name__ == "__main__":
    main()
