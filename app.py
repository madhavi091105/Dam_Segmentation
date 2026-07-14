import streamlit as st
import numpy as np
from PIL import Image

from utils.enhancement import enhance_image
from utils.classification import run_classification
from utils.detection import run_detection
from utils.segmentation import run_segmentation
from utils.severity import classify_severity

st.set_page_config(page_title="Dam Structural Health Monitor", layout="wide")
st.title("🏗️ Dam Structural Damage Detection Pipeline")
st.caption("Classification → Detection (YOLO) → Segmentation (SegFormer) → Severity")

with st.sidebar:
    st.header("Settings")
    seg_backend = st.selectbox("Segmentation model", ["segformer", "unet"], index=0)
    conf_thresh = st.slider("Detection confidence", 0.05, 0.9, 0.25, 0.05)
    iou_thresh = st.slider("Detection IoU", 0.1, 0.9, 0.45, 0.05)
    use_tta = st.checkbox("Use TTA for segmentation", value=True)
    apply_enhancement = st.checkbox("Apply image enhancement", value=True)
    class_conf_threshold = st.slider("Classifier 'Damaged' threshold", 0.1, 0.9, 0.5, 0.05)

uploaded_file = st.file_uploader("Upload dam surface image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    st.subheader("1️⃣ Original Image")
    st.image(image, use_column_width=True)

    working_image = image
    if apply_enhancement:
        with st.spinner("Enhancing image..."):
            working_image = enhance_image(image)
        st.subheader("2️⃣ Enhanced Image")
        st.image(working_image, use_column_width=True)

    st.subheader("3️⃣ Classification — Damaged or Not?")
    with st.spinner("Running classification..."):
        cls_result = run_classification(working_image)

    is_damaged = (cls_result["label"] == "Damaged") and \
                 (cls_result["all_probs"].get("Damaged", 0) >= class_conf_threshold)

    col1, col2 = st.columns([1, 2])
    with col1:
        badge = "🔴 Damaged" if is_damaged else "🟢 Undamaged"
        st.markdown(f"### {badge}")
        st.write(f"Confidence: **{cls_result['confidence']:.2%}**")
    with col2:
        st.json(cls_result["all_probs"])

    if not is_damaged:
        st.success("Classifier says surface looks undamaged — skipping detection & segmentation.")
        st.stop()

    st.subheader("4️⃣ Detection")
    with st.spinner("Running detection..."):
        annotated_img, detections = run_detection(
            working_image, conf=conf_thresh, iou=iou_thresh
        )
    st.image(annotated_img, use_column_width=True)
    if detections:
        st.table(
            [{"Class": d["class_name"], "Confidence": round(d["confidence"], 3)}
             for d in detections]
        )
    else:
        st.info("No individual defects detected by YOLO, segmentation might still catch subtle regions.")

    st.subheader("5️⃣ Segmentation")
    with st.spinner("Running segmentation..."):
        mask, overlay, per_class_pct = run_segmentation(
            working_image, backend=seg_backend, use_tta=use_tta
        )
    col1, col2 = st.columns(2)
    with col1:
        st.image(overlay, caption="Overlay (red=cracks, yellow=spalling)", use_column_width=True)
    with col2:
        st.write("**Pixel-wise class distribution:**")
        st.json(per_class_pct)

    st.subheader("6️⃣ Severity Classification")
    severity = classify_severity(per_class_pct, detections)
    level = severity["severity_level"]
    color = {"Low": "green", "Medium": "orange", "High": "red"}[level]
    st.markdown(f"### Severity: :{color}[{level}]")
    st.json(severity)

else:
    st.info("👆 Upload an image to start the pipeline.")