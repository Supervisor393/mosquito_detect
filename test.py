import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

import config
from dataset import MosquitoDataset, collate_fn
from model import get_model


def test(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Testing"):
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_class": precision_class,
        "recall_class": recall_class,
        "f1_class": f1_class,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["No Mosquito", "Mosquito"])
    plt.yticks(tick_marks, ["No Mosquito", "Mosquito"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test mosquito detection model")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--model_type", default="efficient", choices=["efficient", "simple"], help="Model type")
    parser.add_argument("--data_dir", default=config.DATA_DIR, help="Data directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use for testing")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = get_model(args.model_type, num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")

    test_dataset = MosquitoDataset(os.path.join(args.data_dir, "test"))
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    results = test(model, test_loader, device)

    print("\n=== Test Results ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")

    print("\nClass-wise Results:")
    print(f"  No Mosquito - Precision: {results['precision_class'][0]:.4f}, Recall: {results['recall_class'][0]:.4f}, F1: {results['f1_class'][0]:.4f}")
    print(f"  Mosquito    - Precision: {results['precision_class'][1]:.4f}, Recall: {results['recall_class'][1]:.4f}, F1: {results['f1_class'][1]:.4f}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    cm_path = os.path.join(config.MODEL_DIR, "confusion_matrix.png")
    plot_confusion_matrix(results["confusion_matrix"], cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()