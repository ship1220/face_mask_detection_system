import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

# Expected project files:
# app.py
# requirements.txt
# mask_detector.keras
# face_detector/
#   deploy.prototxt
#   res10_300x300_ssd_iter_140000.caffemodel

IMG_SIZE = (224, 224)
CLASS_NAMES = ["with_mask", "without_mask"]

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mask_detector.keras"
FACE_PROTO = BASE_DIR / "face_detector" / "deploy.prototxt"
FACE_WEIGHTS = BASE_DIR / "face_detector" / "res10_300x300_ssd_iter_140000.caffemodel"

st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #0e0d16; color: #f5f3ff; }
    .block-container { max-width: 1100px; padding-top: 2rem; }
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #211638, #151321);
        border: 1px solid #3a2b58; margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; color: #ffffff; }
    .hero p { color: #c8bfdc; margin-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>😷 Face Mask Detection</h1>
      <p>Real-time webcam detection using OpenCV SSD and a MobileNetV2 classifier.
      Green boxes indicate <b>with mask</b>; red boxes indicate <b>without mask</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How it works"):
    st.write(
        "The webcam sends video frames to the app. OpenCV's SSD detector locates "
        "faces in each frame, and MobileNetV2 classifies each detected face as "
        "'with mask' or 'without mask'. The displayed confidence is the model's "
        "predicted probability, not a guarantee."
    )
    st.caption(
        "Camera access requires browser permission. On Streamlit Cloud, the browser "
        "must be able to establish a WebRTC connection; some restricted networks "
        "may block live video."
    )


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. Put it beside app.py."
        )
    return tf.keras.models.load_model(str(MODEL_PATH))


def load_face_detector():
    missing = [str(p) for p in (FACE_PROTO, FACE_WEIGHTS) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required face-detector files were not found:\n- "
            + "\n- ".join(missing)
            + "\nCreate a real face_detector folder containing both files."
        )
    # OpenCV 4.x includes the Caffe model loader used by this project.
    return cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_WEIGHTS))


def detect_faces(detector, image_bgr, threshold=0.35, padding=0.20):
    h, w = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    detector.setInput(blob)
    detections = detector.forward()
    boxes = []

    for i in range(detections.shape[2]):
        face_confidence = float(detections[0, 0, i, 2])
        if face_confidence < threshold:
            continue

        x1, y1, x2, y2 = (
            detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        ).astype(int)

        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue

        x1 = max(0, int(x1 - padding * box_w))
        y1 = max(0, int(y1 - padding * box_h))
        x2 = min(w - 1, int(x2 + padding * box_w))
        y2 = min(h - 1, int(y2 + padding * box_h))

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2, face_confidence))

    return boxes


def classify_face(model, face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, IMG_SIZE)
    batch = np.expand_dims(face_rgb.astype(np.float32), axis=0)
    # The saved model contains its own rescaling layer. Keep pixels in 0–255 range.
    probabilities = model.predict(batch, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    return CLASS_NAMES[class_index], float(probabilities[class_index])


def annotate_frame(image_bgr, detector, model, threshold):
    output = image_bgr.copy()
    boxes = detect_faces(detector, image_bgr, threshold=threshold)

    for x1, y1, x2, y2, _face_confidence in boxes:
        face_crop = image_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        label, confidence = classify_face(model, face_crop)
        is_mask = label == "with_mask"
        color = (0, 180, 0) if is_mask else (0, 0, 220)
        label_text = f"{label.replace('_', ' ')}: {confidence * 100:.1f}%"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.58, 2
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, font, scale, thickness
        )
        label_top = max(0, y1 - text_h - baseline - 10)
        cv2.rectangle(
            output,
            (x1, label_top),
            (min(output.shape[1] - 1, x1 + text_w + 8), y1),
            color,
            -1,
        )
        cv2.putText(
            output,
            label_text,
            (x1 + 4, max(text_h + 2, y1 - baseline - 5)),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return output, len(boxes)


class MaskDetectionProcessor(VideoProcessorBase):
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
        self.detector = load_face_detector()
        self.lock = threading.Lock()
        self.last_face_count = 0

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")

        # The DNN detector and model are processed one frame at a time per stream.
        with self.lock:
            annotated, face_count = annotate_frame(
                image_bgr, self.detector, self.model, self.threshold
            )
            self.last_face_count = face_count

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


with st.sidebar:
    st.subheader("Detection settings")
    face_threshold = st.slider(
        "Face detection threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.35,
        step=0.05,
        help="Lower values may detect more faces but can introduce false detections.",
    )
    st.caption("Classifier: MobileNetV2")
    st.caption("Face detector: OpenCV SSD")

try:
    model = load_model()
    # Check required files before the user starts the webcam.
    missing_files = [p for p in (FACE_PROTO, FACE_WEIGHTS) if not p.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing face-detector file(s):\n- "
            + "\n- ".join(str(p) for p in missing_files)
            + "\nBoth files must be inside the face_detector folder."
        )
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Live webcam")
st.write("Select **START** below and allow camera access in your browser. Select **STOP** when finished.")

webrtc_streamer(
    key="face-mask-live-stream",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: MaskDetectionProcessor(model, face_threshold),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)

st.caption(
    "This demo classifies faces into two categories only: with mask and without mask. "
    "Live video is processed frame by frame; performance depends on your device and connection."
)
