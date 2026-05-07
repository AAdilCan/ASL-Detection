import pathlib
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_DIR = pathlib.Path("data")
MODEL_PATH = pathlib.Path("model.pkl")
CONFUSION_MATRIX_PATH = pathlib.Path("confusion_matrix.png")


def load_data():
    X, y = [], []
    skipped = []

    for npy_file in sorted(DATA_DIR.glob("*.npy")):
        label = npy_file.stem
        arr = np.load(npy_file)

        if arr.ndim != 2 or arr.shape[1] != 63:
            print(f"Warning: {npy_file.name} has shape {arr.shape}, expected (N, 63) — skipping.")
            skipped.append(label)
            continue

        X.append(arr)
        y.extend([label] * len(arr))

    if not X:
        return np.array([]), np.array([])

    return np.vstack(X), np.array(y)


def main():
    if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.npy")):
        print("No .npy files found in data/. Run python download_dataset.py first.")
        sys.exit(1)

    X, y = load_data()

    if len(X) == 0:
        print("No valid data loaded. Check your .npy files.")
        sys.exit(1)

    labels = sorted(set(y))
    counts = {label: int(np.sum(y == label)) for label in labels}

    print(f"Labels found: {' '.join(labels)}")
    print(f"Samples per label: {counts}")
    print(f"Total samples: {len(y)}")

    if len(labels) < 2:
        print("\nError: Need at least 2 classes to train. Run python download_dataset.py first.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=labels))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cmap="Blues",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("ASL Confusion Matrix")
    # rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print(
        f"Model saved to {MODEL_PATH} | "
        f"Accuracy: {accuracy * 100:.2f}% | "
        "Run python predict.py"
    )


if __name__ == "__main__":
    main()
