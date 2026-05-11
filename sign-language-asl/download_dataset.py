import os
import sys
import subprocess
import pathlib
import csv

KAGGLE_JSON_PATH = pathlib.Path.home() / ".kaggle" / "kaggle.json"
KAGGLE_DATASET = "grassknoted/asl-alphabet"
KAGGLE_DATA_DIR = pathlib.Path("kaggle_data")
DATA_DIR = pathlib.Path("data")
SKIP_LABELS = {"J", "Z"}
MIN_SAMPLES = 50
MAX_IMAGES_PER_LABEL = 300  # cap to keep processing time reasonable (~5 min total)


def print_setup_instructions():
    print("=== Kaggle Setup ===")
    print("1. Go to kaggle.com → Account → Create New API Token")
    print("2. Place kaggle.json at ~/.kaggle/kaggle.json (Linux/Mac)")
    print("   or C:\\Users\\YOU\\.kaggle\\kaggle.json (Windows)")
    print("3. Run: python download_dataset.py")
    print("====================")


def ensure_kaggle_installed():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Installing kaggle package...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaggle", "-q"],
            check=True,
        )


def download_dataset():
    print(f"Downloading dataset: {KAGGLE_DATASET}")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "--path", str(KAGGLE_DATA_DIR),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Download failed. Check your internet connection and Kaggle credentials.")
        print(f"Error: {result.stderr.strip()}")
        print(f"You can also download manually from: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        sys.exit(1)
    print("Download complete.")


def inspect_kaggle_data():
    print("\n=== Contents of kaggle_data/ ===")
    for root, dirs, files in os.walk(KAGGLE_DATA_DIR):
        level = root.replace(str(KAGGLE_DATA_DIR), "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        for f in files[:5]:
            print(f"{sub_indent}{f}")
        if len(files) > 5:
            print(f"{sub_indent}... ({len(files)} files total)")
    print("================================\n")


def find_csv_files():
    csv_files = list(KAGGLE_DATA_DIR.rglob("*.csv"))
    return csv_files


def find_image_folders():
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    label_folders = {}
    for item in KAGGLE_DATA_DIR.rglob("*"):
        if item.is_dir():
            images = [f for f in item.iterdir() if f.suffix.lower() in image_extensions]
            if images:
                label = item.name.upper()
                if label not in SKIP_LABELS:
                    # cap images per label to avoid multi-hour processing
                    label_folders[label] = images[:MAX_IMAGES_PER_LABEL]
    return label_folders


def process_csv_files(csv_files):
    label_data = {}
    for csv_path in csv_files:
        label = csv_path.stem.upper()
        if label in SKIP_LABELS:
            print(f"Skipping {label} (movement-based sign)")
            continue

        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                # skip header rows that contain non-numeric values
                try:
                    floats = [float(v) for v in row]
                except ValueError:
                    continue
                if len(floats) == 63:
                    rows.append(floats)
                else:
                    print(f"  Warning: row in {csv_path.name} has {len(row)} columns, expected 63 — skipping row")

        if len(rows) < MIN_SAMPLES:
            print(f"Warning: {label} has only {len(rows)} valid samples (< {MIN_SAMPLES}), skipping.")
            continue

        label_data[label] = rows
        print(f"  CSV: {label} — {len(rows)} samples")

    return label_data


def process_image_folders(label_folders):
    import mediapipe as mp
    import cv2
    import numpy as np

    mp_hands = mp.solutions.hands
    label_data = {}

    for label, image_paths in sorted(label_folders.items()):
        if label in SKIP_LABELS:
            print(f"Skipping {label} (movement-based sign)")
            continue

        landmarks_list = []
        total = len(image_paths)

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        ) as hands:
            for idx, img_path in enumerate(image_paths):
                if (idx + 1) % 10 == 0 or idx == 0:
                    print(f"Processing {label}: {idx + 1}/{total} images...", end="\r")

                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                # MediaPipe expects RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    coords = []
                    for lm in hand.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    if len(coords) == 63:
                        landmarks_list.append(coords)

        print()  # newline after progress

        if len(landmarks_list) == 0:
            print(f"Warning: {label} produced 0 MediaPipe detections — skipping.")
            continue

        if len(landmarks_list) < MIN_SAMPLES:
            print(f"Warning: {label} has only {len(landmarks_list)} valid samples (< {MIN_SAMPLES}), skipping.")
            continue

        label_data[label] = landmarks_list
        print(f"  Images: {label} — {len(landmarks_list)} samples")

    return label_data


def save_npy_files(label_data):
    import numpy as np

    DATA_DIR.mkdir(exist_ok=True)
    for label, samples in label_data.items():
        out_path = DATA_DIR / f"{label}.npy"
        arr = np.array(samples, dtype=np.float32)
        np.save(out_path, arr)

    print(f"\nSaved {len(label_data)} label files to {DATA_DIR}/")


def check_existing_data():
    existing = list(DATA_DIR.glob("*.npy"))
    if existing:
        answer = input(f"data/ already has {len(existing)} .npy files. Overwrite? (y/n): ").strip().lower()
        if answer != "y":
            print("Aborted. Existing data preserved.")
            sys.exit(0)


def main():
    print_setup_instructions()

    if not KAGGLE_JSON_PATH.exists():
        print(f"\nError: {KAGGLE_JSON_PATH} not found.")
        print_setup_instructions()
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)
    KAGGLE_DATA_DIR.mkdir(exist_ok=True)

    check_existing_data()
    ensure_kaggle_installed()
    download_dataset()
    inspect_kaggle_data()

    csv_files = find_csv_files()
    label_data = {}

    if csv_files:
        print(f"Found {len(csv_files)} CSV file(s). Processing as landmark CSVs...")
        label_data = process_csv_files(csv_files)
    else:
        print("No CSVs found. Looking for image folders...")
        label_folders = find_image_folders()
        if not label_folders:
            print("Error: Could not find CSV files or image folders in kaggle_data/.")
            sys.exit(1)
        label_data = process_image_folders(label_folders)

    if not label_data:
        print("Error: No valid label data could be extracted.")
        sys.exit(1)

    save_npy_files(label_data)

    labels_str = " ".join(sorted(label_data.keys()))
    counts = {k: len(v) for k, v in label_data.items()}
    total = sum(counts.values())

    print("\n=== Dataset Ready ===")
    print(f"Labels processed: {labels_str}")
    print(f"Samples per label: {counts}")
    print(f"Total samples: {total}")
    print("Next step: python train.py")


if __name__ == "__main__":
    main()
