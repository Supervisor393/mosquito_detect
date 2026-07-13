import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import config


def create_data_structure(base_dir):
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(dir_path, "mosquito"), exist_ok=True)
        os.makedirs(os.path.join(dir_path, "no_mosquito"), exist_ok=True)

    return train_dir, val_dir, test_dir


def prepare_humbugdb(metadata_path, audio_dir, output_dir, test_size=0.2, val_size=0.2):
    df = pd.read_csv(metadata_path)
    print(f"Total samples in metadata: {len(df)}")

    df["is_mosquito"] = (df["sound_type"] == "mosquito").astype(int)
    print(f"Class distribution:")
    print(df["is_mosquito"].value_counts())

    valid_samples = []
    for _, row in df.iterrows():
        audio_path = os.path.join(audio_dir, f"{row['id']}.wav")
        if os.path.exists(audio_path):
            valid_samples.append((audio_path, row["is_mosquito"], row["sound_type"]))
        else:
            print(f"Warning: Audio file not found: {audio_path}")

    print(f"Valid samples with audio files: {len(valid_samples)}")

    features = [s[0] for s in valid_samples]
    labels = [s[1] for s in valid_samples]

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels, test_size=val_size / (1 - test_size), random_state=42, stratify=train_labels
    )

    train_dir, val_dir, test_dir = create_data_structure(output_dir)

    def copy_files(features, labels, dest_dir):
        mosquito_count = 0
        no_mosquito_count = 0
        for feature, label in zip(features, labels):
            file_name = os.path.basename(feature)
            if label == 1:
                dest = os.path.join(dest_dir, "mosquito", file_name)
                mosquito_count += 1
            else:
                dest = os.path.join(dest_dir, "no_mosquito", file_name)
                no_mosquito_count += 1
            if not os.path.exists(dest):
                os.symlink(feature, dest)
        return mosquito_count, no_mosquito_count

    train_mosquito, train_no_mosquito = copy_files(train_features, train_labels, train_dir)
    val_mosquito, val_no_mosquito = copy_files(val_features, val_labels, val_dir)
    test_mosquito, test_no_mosquito = copy_files(test_features, test_labels, test_dir)

    print(f"\nData prepared successfully!")
    print(f"\nTrain set:")
    print(f"  Mosquito: {train_mosquito}")
    print(f"  No Mosquito: {train_no_mosquito}")
    print(f"\nVal set:")
    print(f"  Mosquito: {val_mosquito}")
    print(f"  No Mosquito: {val_no_mosquito}")
    print(f"\nTest set:")
    print(f"  Mosquito: {test_mosquito}")
    print(f"  No Mosquito: {test_no_mosquito}")


def prepare_custom_data(source_dir, train_dir, val_dir, test_dir, test_size=0.2, val_size=0.2):
    for label in ["mosquito", "no_mosquito"]:
        source_label_dir = os.path.join(source_dir, label)
        if not os.path.exists(source_label_dir):
            continue

        files = [f for f in os.listdir(source_label_dir) if f.endswith(".wav")]

        train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=val_size / (1 - test_size), random_state=42)

        for f in train_files:
            src = os.path.join(source_label_dir, f)
            dst = os.path.join(train_dir, label, f)
            if not os.path.exists(dst):
                os.symlink(src, dst)

        for f in val_files:
            src = os.path.join(source_label_dir, f)
            dst = os.path.join(val_dir, label, f)
            if not os.path.exists(dst):
                os.symlink(src, dst)

        for f in test_files:
            src = os.path.join(source_label_dir, f)
            dst = os.path.join(test_dir, label, f)
            if not os.path.exists(dst):
                os.symlink(src, dst)

        print(f"Label '{label}':")
        print(f"  Train: {len(train_files)} files")
        print(f"  Val: {len(val_files)} files")
        print(f"  Test: {len(test_files)} files")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for mosquito detection")
    parser.add_argument("--mode", default="humbugdb", choices=["humbugdb", "custom"], 
                        help="Data preparation mode")
    parser.add_argument("--source_dir", help="Source directory for custom data mode")
    parser.add_argument("--output_dir", default=config.DATA_DIR, help="Output directory for split data")
    parser.add_argument("--metadata_path", default=config.HUMBUGDB_METADATA_PATH, 
                        help="Path to HumbugDB metadata CSV")
    parser.add_argument("--audio_dir", default=config.HUMBUGDB_AUDIO_DIR, 
                        help="Directory containing audio files")
    args = parser.parse_args()

    if args.mode == "humbugdb":
        if not os.path.exists(args.metadata_path):
            print(f"Error: Metadata file not found at {args.metadata_path}")
            return
        if not os.path.exists(args.audio_dir):
            print(f"Error: Audio directory not found at {args.audio_dir}")
            return
        prepare_humbugdb(args.metadata_path, args.audio_dir, args.output_dir)
    else:
        if not args.source_dir:
            print("Error: --source_dir is required for custom mode")
            return
        train_dir, val_dir, test_dir = create_data_structure(args.output_dir)
        prepare_custom_data(args.source_dir, train_dir, val_dir, test_dir)


if __name__ == "__main__":
    main()