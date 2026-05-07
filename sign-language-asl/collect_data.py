import sys
import pathlib
import numpy as np
import cv2
import mediapipe as mp

SUPPORTED_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
    "YES", "NO", "HELLO", "PLEASE", "SORRY", "THANK_YOU",
]

TARGET_SAMPLES = 200
DATA_DIR = pathlib.Path("data")


def load_existing(label):
    path = DATA_DIR / f"{label}.npy"
    if path.exists():
        arr = np.load(path)
        print(f"Loaded {len(arr)} existing samples for '{label}'.")
        return arr.tolist()
    return []


def save_samples(label, samples):
    DATA_DIR.mkdir(exist_ok=True)
    arr = np.array(samples, dtype=np.float32)
    np.save(DATA_DIR / f"{label}.npy", arr)
    print(f"Saved {len(samples)} samples for '{label}' to data/{label}.npy")


def main():
    label = sys.argv[1].upper() if len(sys.argv) > 1 else "A"

    if label not in SUPPORTED_LABELS:
        print(f"Error: '{label}' is not a supported label.")
        print(f"Supported: {', '.join(SUPPORTED_LABELS)}")
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)
    samples = load_existing(label)
    collecting = False

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        hands.close()
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from webcam.")
                continue

            # mirror the frame so movement feels natural
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if collecting and len(samples) < TARGET_SAMPLES:
                    hand = results.multi_hand_landmarks[0]
                    coords = []
                    for lm in hand.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    if len(coords) == 63:
                        samples.append(coords)

            count = len(samples)
            status = "COLLECTING" if collecting else "SPACE=start"
            cv2.putText(
                frame,
                f"Label: {label} | Collected: {count}/{TARGET_SAMPLES} | {status}  Q=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if collecting else (255, 255, 255),
                2,
            )

            cv2.imshow("ASL Data Collector", frame)

            if count >= TARGET_SAMPLES:
                save_samples(label, samples)
                print(f"Reached {TARGET_SAMPLES} samples for '{label}'. Exiting.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                collecting = True
                print("Started collecting...")
            elif key == ord("q"):
                if samples:
                    save_samples(label, samples)
                else:
                    print("No samples collected.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
