import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Expected project files:
# app.py
# mask_detector.keras
# face_detector/
#   deploy.prototxt
#   res10_300x300_ssd_iter_140000.caffemodel

IMG_SIZE = (224, 224)
CLASS_NAMES = ["with_mask", "without_mask"]

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mask_detector.keras"
FACE_PROTO = BASE_DIR / "face_detector" / "deploy.prototxt"
FACE_WEIGHTS = (
    BASE_DIR / "face_detector" / "res10_300x300_ssd_iter_140000.caffemodel"
)

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
)

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
      <p>Capture a webcam photo to detect faces and classify each as
      <b>with mask</b> or <b>without mask</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How it works"):
    st.write(
        "OpenCV's SSD face detector locates faces in the captured image. "
        "The MobileNetV2 CNN then classifies each detected face into one of "
        "two classes: with_mask or without_mask. The displayed confidence is "
        "the classifier's predicted probability, not a guarantee."
    )


@st.cache_resource
def load_models():
    missing = [
        str(p) for p in [MODEL_PATH, FACE_PROTO, FACE_WEIGHTS] if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required model files were not found:\n- "
            + "\n- ".join(missing)
            + "\nPlace mask_detector.keras and the face_detector folder beside app.py."
        )

    model = tf.keras.models.load_model(str(MODEL_PATH))
    detector = cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_WEIGHTS))
    return model, detector


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
    # The saved model contains its own MobileNetV2 rescaling layer.
    # Pass RGB pixels in the 0–255 range; do not divide by 255 here.
    probabilities = model.predict(batch, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    return CLASS_NAMES[class_index], float(probabilities[class_index])


def annotate_image(image_bgr, boxes, model):
    output = image_bgr.copy()
    detections = []

    for x1, y1, x2, y2, face_confidence in boxes:
        label, confidence = classify_face(model, image_bgr[y1:y2, x1:x2])
        color = (0, 180, 0) if label == "with_mask" else (0, 0, 220)
        label_text = f"{label.replace('_', ' ')}: {confidence * 100:.1f}%"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.6, 2
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, font, scale, thickness
        )
        text_x = max(0, min(x1, output.shape[1] - text_w - 8))
        text_y = y1 - 8 if y1 - text_h - baseline - 8 >= 0 else min(
            output.shape[0] - baseline - 3, y1 + text_h + baseline + 8
        )
        bg_top = max(0, text_y - text_h - baseline - 4)
        bg_bottom = min(output.shape[0] - 1, text_y + baseline + 2)
        cv2.rectangle(
            output,
            (text_x, bg_top),
            (min(output.shape[1] - 1, text_x + text_w + 8), bg_bottom),
            color,
            -1,
        )
        cv2.putText(
            output,
            label_text,
            (text_x + 4, text_y - 2),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        detections.append(
            {
                "Face": len(detections) + 1,
                "Prediction": label.replace("_", " "),
                "Confidence": f"{confidence * 100:.1f}%",
            }
        )

    return output, detections


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
    st.caption("Model: MobileNetV2 · Face detector: OpenCV SSD")

try:
    model, detector = load_models()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Capture a webcam image")
photo = st.camera_input("Take a photo with your webcam")

if photo is not None:
    pil_image = Image.open(photo).convert("RGB")
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("Detecting faces and classifying mask status..."):
        boxes = detect_faces(detector, image_bgr, threshold=face_threshold)

        if boxes:
            annotated_bgr, detections = annotate_image(image_bgr, boxes, model)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            st.success(f"Processed image. Detected {len(detections)} face(s).")
            st.image(annotated_rgb, caption="Mask detection result", use_container_width=True)
            st.subheader("Predictions")
            st.dataframe(detections, use_container_width=True, hide_index=True)
        else:
            st.warning(
                "No faces were detected. Try taking another photo with your face "
                "clearly visible, facing the camera, and with better lighting."
            )
            st.image(image_rgb, caption="Captured image", use_container_width=True)

st.caption(
    "This demo classifies only two categories: with mask and without mask. "
    "It processes a captured image rather than continuous live video."
)
