import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import io
from gtts import gTTS
from collections import Counter, deque

# =========================================================
# PYTORCH SAFE LOAD FIX (PYTORCH 2.6+)
# =========================================================
import torch
import ultralytics.nn.tasks
import torch.nn.modules.container
import torch.nn.modules.conv
import torch.nn.modules.batchnorm
import torch.nn.modules.activation

SAFE_GLOBALS = [
    ultralytics.nn.tasks.DetectionModel,
    torch.nn.modules.container.Sequential,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.activation.ReLU,
]

try:
    torch.serialization.add_safe_globals(SAFE_GLOBALS)
except Exception:
    pass

# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Vision Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if "stop_camera" not in st.session_state:
    st.session_state.stop_camera = False

if "scene_buffer" not in st.session_state:
    st.session_state.scene_buffer = deque(maxlen=12)

if "last_spoken_time" not in st.session_state:
    st.session_state.last_spoken_time = 0

if "last_spoken_text" not in st.session_state:
    st.session_state.last_spoken_text = ""

if "last_process_time" not in st.session_state:
    st.session_state.last_process_time = time.time()

# =========================================================
# SPEECH CONFIGURATION
# =========================================================
SPEECH_INTERVAL = 15  # seconds

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_object_model():
    return YOLO("yolov8s.pt")

@st.cache_resource
def load_emotion_model():
    try:
        return YOLO("emotion_detection_model.pt")
    except Exception as e:
        st.error(f"❌ Emotion model load failed: {e}")
        return None

object_model = load_object_model()
emotion_model = load_emotion_model()

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_position(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if cx < w / 3:
        horizontal = "left"
    elif cx > 2 * w / 3:
        horizontal = "right"
    else:
        horizontal = "center"

    if cy < h / 3:
        vertical = "top"
    elif cy > 2 * h / 3:
        vertical = "bottom"
    else:
        vertical = "middle"

    return f"{vertical} {horizontal}"

def generate_audio(text):
    tts = gTTS(text=text, lang="en")
    audio = io.BytesIO()
    tts.write_to_fp(audio)
    audio.seek(0)
    return audio

# =========================================================
# STABLE SCENE BUILDER
# =========================================================
def build_stable_scene():
    object_phrases = []
    emotion_labels = []

    for frame in st.session_state.scene_buffer:
        object_phrases.extend(frame["objects"])
        emotion_labels.extend(frame["emotions"])

    obj_counts = Counter(object_phrases)
    emo_counts = Counter(emotion_labels)

    speech = []

    for obj, count in obj_counts.items():
        if count >= 4:
            speech.append(obj)

    if emo_counts:
        emo, freq = emo_counts.most_common(1)[0]
        if freq >= 4:
            speech.append(f"The person appears {emo}")

    if speech:
        return "Scene description: " + ". ".join(speech)

    return None

# =========================================================
# IMAGE PROCESSING
# =========================================================
def process_image(image):
    result = object_model(image, verbose=False)[0]
    emo_res = emotion_model(image, verbose=False)[0] if emotion_model else None

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    w, h = image.size
    frame_objects = []
    frame_emotions = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = object_model.names[int(box.cls[0])]
        pos = get_position(x1, y1, x2, y2, w, h)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 15), label, fill="red", font=font)

        frame_objects.append(f"{label} at {pos}")

    if emo_res:
        for box in emo_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            emo = emotion_model.names[int(box.cls[0])]

            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
            draw.text((x1, y2 + 5), emo, fill="blue", font=font)

            frame_emotions.append(emo)

    st.session_state.scene_buffer.append({
        "objects": frame_objects,
        "emotions": frame_emotions
    })

    return annotated, build_stable_scene()

# =========================================================
# HOME PAGE
# =========================================================
def home_page():
    st.title("👁️ Vision Assistant for Visually Impaired")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🎯 Features")
        st.write("""
        - Real-time Object Detection  
        - Emotion Recognition  
        - Position Awareness  
        - Stable Scene Understanding  
        - Continuous Audio Feedback  
        """)

    with c2:
        st.markdown("### 📱 How to Use")
        st.write("""
        1. Upload an image for detection  
        2. Start live camera for real-time analysis  
        3. Listen to automatic audio descriptions  
        """)

    st.markdown("---")

    m1, m2, m3 = st.columns(3)
    m1.metric("Object Model", "YOLOv8s")
    m2.metric("Emotion Model", "Custom YOLOv8")
    m3.metric("Audio System", "gTTS Enabled")

# =========================================================
# UPLOAD IMAGE PAGE
# =========================================================
def upload_page():
    st.title("📤 Upload Image")
    st.markdown("---")

    f = st.file_uploader("Choose an image", ["jpg", "jpeg", "png", "webp"])
    if f:
        img = Image.open(f).convert("RGB")
        annotated, speech = process_image(img)

        c1, c2 = st.columns(2)
        with c1:
            st.image(img, caption="Original Image", use_container_width=True)
        with c2:
            st.image(annotated, caption="Detection Results", use_container_width=True)

        st.markdown("## 🔊 Audio Description")
        st.write(speech)

        if speech:
            st.audio(generate_audio(speech), autoplay=True)

# =========================================================
# LIVE CAMERA PAGE
# =========================================================
def live_camera_page():
    st.title("📹 Live Camera Detection")
    st.markdown("---")

    c1, c2 = st.columns(2)

    if c1.button("🎥 Start Live Detection", type="primary"):
        st.session_state.camera_active = True
        st.session_state.stop_camera = False
        st.session_state.scene_buffer.clear()
        st.session_state.last_spoken_time = 0
        st.session_state.last_spoken_text = ""

    if c2.button("⏹️ Stop Detection"):
        st.session_state.camera_active = False
        st.session_state.stop_camera = True

    if not st.session_state.camera_active:
        st.info("Camera is stopped.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # ===== CRITICAL LAG FIX =====
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    video = st.empty()
    text = st.empty()
    audio_box = st.empty()

    while st.session_state.camera_active and not st.session_state.stop_camera:

        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated, speech = process_image(Image.fromarray(rgb))

        video.image(annotated, use_container_width=True)
        text.markdown(f"### 🔊 Stable Scene Description\n{speech}")

        now = time.time()

        if speech and (now - st.session_state.last_spoken_time >= SPEECH_INTERVAL):
            if speech != st.session_state.last_spoken_text:
                audio_box.audio(generate_audio(speech), autoplay=True)
                st.session_state.last_spoken_time = now
                st.session_state.last_spoken_text = speech

        time.sleep(0.01)

    cap.release()
    st.success("Camera stopped successfully.")

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/visible.png", width=80)

    st.markdown("## Navigation")
    page = st.radio("Go to:", ["🏠 Home", "📤 Upload Image", "📹 Live Camera"])

    st.markdown("---")
    st.markdown("## Settings")
    st.checkbox("Enable Audio", value=True)

    st.markdown("---")
    st.markdown("## Model Info")
    st.write("Object Model: YOLOv8s")
    st.write("Emotion Model: emotion_detection_model.pt")

# =========================================================
# ROUTING
# =========================================================
if page == "🏠 Home":
    home_page()
elif page == "📤 Upload Image":
    upload_page()
elif page == "📹 Live Camera":
    live_camera_page()

st.markdown("---")
st.markdown("👁️ Vision Assistant | Object + Emotion Detection | Stable Audio Feedback")