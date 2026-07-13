import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config
from dataset import get_dataloader, AudioAugmentation, mixup_data, mixup_criterion
from model import get_model


def train_one_epoch(model, dataloader, criterion, optimizer, device, mixup_alpha=0.2):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for features, labels in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if mixup_alpha > 0 and np.random.random() < 0.5:
            features, labels_a, labels_b, lam = mixup_data(features, labels, mixup_alpha)
            outputs = model(features)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        else:
            outputs = model(features)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")

    return avg_loss, accuracy, precision, recall, f1


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validating"):
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")

    return avg_loss, accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Train mosquito detection model")
    parser.add_argument("--model", default="efficient", choices=["efficient", "simple"], help="Model type")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--data_dir", default=config.DATA_DIR, help="Data directory")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha parameter")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use for training")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    train_transform = AudioAugmentation(p=0.5)
    train_loader = get_dataloader(os.path.join(args.data_dir, "train"), batch_size=args.batch_size, transform=train_transform, is_train=True)
    val_loader = get_dataloader(os.path.join(args.data_dir, "val"), batch_size=args.batch_size, is_train=False)

    if len(train_loader.dataset) == 0:
        print(f"Error: No training samples found in {os.path.join(args.data_dir, 'train')}")
        print("Please prepare data first using: python prepare_data.py --source_dir <your_data_dir>")
        return

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = get_model(args.model, num_classes=config.NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, mixup_alpha=args.mixup_alpha
        )
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")

        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model_path = os.path.join(config.MODEL_DIR, f"mosquito_detector_{args.model}_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break

    model_path = os.path.join(config.MODEL_DIR, f"mosquito_detector_{args.model}_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")


if __name__ == "__main__":
    main()