import collections
import pathlib
import pickle
import sys

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = pathlib.Path("model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        print("Run python train.py first")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def get_color(confidence):
    if confidence >= 0.85:
        return (0, 200, 0)      # green
    elif confidence >= 0.60:
        return (0, 165, 255)    # orange
    return (0, 0, 220)          # red


def main():
    clf = load_model()
    class_labels = list(clf.classes_)
    labels_str = " ".join(class_labels)
    print(f"Loaded model with {len(class_labels)} classes: {labels_str}")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        hands.close()
        sys.exit(1)

    # keep last 10 predictions for majority-vote smoothing
    vote_buffer = collections.deque(maxlen=10)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from webcam.")
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            prediction = ""
            confidence = 0.0

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                coords = []
                for lm in hand.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                if len(coords) == 63:
                    features = np.array(coords, dtype=np.float32).reshape(1, -1)
                    proba = clf.predict_proba(features)[0]
                    top_idx = int(np.argmax(proba))
                    vote_buffer.append(top_idx)

                    # majority vote across buffer
                    most_common_idx = collections.Counter(vote_buffer).most_common(1)[0][0]
                    prediction = class_labels[most_common_idx]
                    confidence = float(proba[most_common_idx])

            h, w = frame.shape[:2]

            if prediction:
                color = get_color(confidence)
                cv2.putText(
                    frame, prediction,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                    color, 4,
                )
                cv2.putText(
                    frame, f"{confidence:.0%}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    color, 2,
                )

            # bottom bar with labels and quit hint
            bar_height = 36
            cv2.rectangle(frame, (0, h - bar_height), (w, h), (30, 30, 30), -1)
            bottom_text = f"Q: quit | Labels: {labels_str}"
            cv2.putText(
                frame, bottom_text,
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1,
            )

            cv2.imshow("ASL Predictor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
