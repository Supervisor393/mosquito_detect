# OOD test set

This directory contains a small, reproducible test set for binary mosquito
sound detection. It is intentionally excluded from model training and
threshold selection.

## Composition

- `mosquito/`: 30 recordings from **MosquitoSound (Wingbeats)**, five from
  each of six mosquito species. The original signals were captured by an
  infrared optical sensor at 6 kHz, so they differ substantially from the
  microphone recordings in HumBugDB. The source dataset is Public Domain.
- `no_mosquito/`: 50 recordings from **ESC-50**, one recording from each of
  the five official folds for ten likely false-alarm categories: insects,
  crickets, wind, engine, car horn, train, laughing, coughing, church bells,
  and airplane. ESC-50 is distributed under CC BY-NC.
- `manifest.csv`: per-file ground truth, original identifier, class, source
  URL, sample rate, license, and the reason the sample is considered OOD.
- `source_metadata/`: source labels and metadata needed to reproduce the
  selection.

Sources:

- MosquitoSound: https://huggingface.co/datasets/monster-monash/MosquitoSound
- ESC-50: https://github.com/karolpiczak/ESC-50

## Reproduce and evaluate

From the `mosquito_detect` directory:

```powershell
.venv\Scripts\python.exe download_ood_data.py --seed 2026
.venv\Scripts\python.exe evaluate_ood.py `
  --model_path models\mosquito_detector_efficient_best.pth
```

Report mosquito recall (sensitivity) and the false-positive rate on the
background recordings separately. Do not tune a threshold on these test
samples; select it using the in-distribution validation set or a separate OOD
development set.

## Limitation

The positive recordings are clean optical-sensor signals rather than phone
microphone recordings. They are useful for measuring cross-sensor domain
shift, but they do not replace a small set of genuine phone recordings made
in everyday conditions. Add phone recordings as a separately named source in
`manifest.csv` before the final coursework evaluation.
