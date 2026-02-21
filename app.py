import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from gtts import gTTS
import io
import mediapipe as mp

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="ISL Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1E88E5;
    font-size: 3rem;
    font-weight: bold;
    font-family: 'Segoe UI', sans-serif;
    text-shadow: 2px 2px #888888;
}
.prediction-box {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    font-size: 4rem;
    font-weight: bold;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
}
.confidence-box {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    font-size: 1.2rem;
    text-align: center;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
}
.sidebar-header {
    font-size: 1.5rem;
    color: #764ba2;
    font-weight: bold;
}
.sidebar-notes {
    font-size: 0.95rem;
    color: #555555;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================
st.markdown('<div class="main-header">🤟 ISL Sign Language Recognition</div>', unsafe_allow_html=True)

# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    model = keras.models.load_model("isl_model.h5")
    labels = np.load("isl_labels.npy", allow_pickle=True)
    return model, labels

model, labels = load_model()

# =========================
# Sidebar
# =========================
st.sidebar.markdown('<div class="sidebar-header">⚙️ Settings & Notes</div>', unsafe_allow_html=True)

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
enable_audio = st.sidebar.checkbox("Enable Voice Output", True)

# ✅ CAMERA SELECTOR (ADDED)
camera_mode = st.sidebar.radio(
    "📷 Camera Mode",
    ["Laptop / PC Camera", "Mobile Camera (Front)", "Mobile Camera (Back)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="sidebar-notes">
 📖 How to Use
1. Click Start Camera
2. Show one hand sign
3. Hold steady
4. Letter updates in same box
5. Click Exit App to stop

### ℹ️ Notes:
- This app detects hand signs for **A-Z alphabets and numbers 1-9**.
- Only **one hand** is detected at a time.
- Ensure **good lighting** for better accuracy.
- App works best on PC or mobile browser on same Wi-Fi.
</div>
""", unsafe_allow_html=True)

# =========================
# Layout
# =========================
col1, col2 = st.columns([3, 2])

with col1:
    start = st.button("▶ Start Camera")
    exit_app = st.button("Exit App")
    frame_area = st.empty()

with col2:
    prediction_area = st.empty()
    confidence_area = st.empty()
    audio_area = st.empty()

# =========================
# Session State
# =========================
if "running" not in st.session_state:
    st.session_state.running = False

if "last_letter" not in st.session_state:
    st.session_state.last_letter = ""

# =========================
# MediaPipe
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# =========================
# Button Logic
# =========================
if start:
    st.session_state.running = True

if exit_app:
    st.session_state.running = False
    st.stop()

# =========================
# Camera Setup
# =========================
if st.session_state.running:
    if camera_mode == "Laptop / PC Camera":
        cap = cv2.VideoCapture(0)
    else:
        cap = None
        cam = st.camera_input(
            "Camera",
            key="mobile_cam",
            help="Use back camera" if "Back" in camera_mode else "Use front camera"
        )

    while st.session_state.running:

        # ===== MOBILE CAMERA =====
        if camera_mode != "Laptop / PC Camera":
            if cam is None:
                st.warning("Please enable camera")
                break

            file_bytes = np.asarray(bytearray(cam.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if "Front" in camera_mode:
                frame = cv2.flip(frame, 1)

        # ===== PC CAMERA =====
        else:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        landmarks = []

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        if len(landmarks) == 63:
            preds = model.predict(np.array([landmarks]), verbose=0)[0]
            idx = np.argmax(preds)
            conf = float(preds[idx])

            if conf >= confidence_threshold:
                letter = str(labels[idx])

                if letter != st.session_state.last_letter:
                    st.session_state.last_letter = letter

                    prediction_area.markdown(
                        f'<div class="prediction-box">{letter}</div>',
                        unsafe_allow_html=True
                    )

                    confidence_area.markdown(
                        f'<div class="confidence-box">Confidence: {conf*100:.2f}%</div>',
                        unsafe_allow_html=True
                    )

                    if enable_audio:
                        tts = gTTS(letter)
                        audio = io.BytesIO()
                        tts.write_to_fp(audio)
                        audio_area.audio(audio.getvalue(), format="audio/mp3")

        frame_area.image(frame, channels="BGR")

    if cap:
        cap.release()