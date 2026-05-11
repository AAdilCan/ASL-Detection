import collections
import pathlib
import pickle

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = pathlib.Path(__file__).parent / "model.pkl"

SIGN_DESCRIPTIONS = {
    "A": "Letter A", "B": "Letter B", "C": "Letter C", "D": "Letter D",
    "E": "Letter E", "F": "Letter F", "G": "Letter G", "H": "Letter H",
    "I": "Letter I", "K": "Letter K", "L": "Letter L", "M": "Letter M",
    "N": "Letter N", "O": "Letter O", "P": "Letter P", "Q": "Letter Q",
    "R": "Letter R", "S": "Letter S", "T": "Letter T", "U": "Letter U",
    "V": "Letter V", "W": "Letter W", "X": "Letter X", "Y": "Letter Y",
    "YES": "Word: YES", "NO": "Word: NO", "HELLO": "Word: HELLO",
    "PLEASE": "Word: PLEASE", "SORRY": "Word: SORRY", "THANK_YOU": "Word: THANK YOU",
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
    )


def extract_landmarks(image_rgb, hands):
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32) if len(coords) == 63 else None


def confidence_color(confidence):
    if confidence >= 0.85:
        return "green"
    elif confidence >= 0.60:
        return "orange"
    return "red"


def main():
    st.set_page_config(page_title="ASL Detector", page_icon="🤟", layout="centered")

    st.title("🤟 ASL Sign Language Detector")
    st.caption("Show a hand sign to your camera — supports A–Y (no J/Z) plus YES, NO, HELLO, PLEASE, SORRY, THANK YOU")

    clf = load_model()
    hands = load_hands()
    class_labels = list(clf.classes_)

    if "vote_buffer" not in st.session_state:
        st.session_state.vote_buffer = collections.deque(maxlen=10)

    col_cam, col_result = st.columns([3, 2])

    with col_cam:
        frame = st.camera_input("Point your camera at your hand")

    with col_result:
        st.subheader("Prediction")

        if frame is not None:
            img = Image.open(frame).convert("RGB")
            img_np = np.array(img)

            features = extract_landmarks(img_np, hands)

            if features is not None:
                proba = clf.predict_proba(features.reshape(1, -1))[0]
                top_idx = int(np.argmax(proba))
                st.session_state.vote_buffer.append(top_idx)

                smoothed_idx = collections.Counter(st.session_state.vote_buffer).most_common(1)[0][0]
                prediction = class_labels[smoothed_idx]
                confidence = float(proba[smoothed_idx])
                color = confidence_color(confidence)
                desc = SIGN_DESCRIPTIONS.get(prediction, prediction)

                st.markdown(
                    f"<h1 style='color:{color}; font-size:72px; margin:0'>{prediction}</h1>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{desc}**")
                st.progress(confidence)
                st.caption(f"Confidence: {confidence:.0%}")

                if confidence >= 0.85:
                    st.success("High confidence")
                elif confidence >= 0.60:
                    st.warning("Medium confidence")
                else:
                    st.error("Low confidence — try adjusting your hand position")
            else:
                st.session_state.vote_buffer.clear()
                st.info("No hand detected. Make sure your hand is visible and well-lit.")
        else:
            st.info("Take a photo to get a prediction.")

    with st.expander("Supported signs"):
        cols = st.columns(6)
        for i, label in enumerate(sorted(class_labels)):
            cols[i % 6].markdown(f"**{label}**")


if __name__ == "__main__":
    main()
