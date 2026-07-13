"""
GeoQuery — Streamlit chat app for natural language understanding of RGB imagery.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import time

# Some ops aren't yet implemented on Apple's MPS backend. This makes torch
# fall back to CPU for just those ops instead of crashing — must be set
# before torch is imported anywhere.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import streamlit as st
from PIL import Image

from config import TARGET_CLASSES, MAX_RESOLUTION
from utils.image_utils import (
    validate_extension, load_and_prepare, draw_detections, get_device,
    crop_box, dominant_color_name, relative_size_label,
)
from utils.intent_router import classify
from utils.report import build_pdf_report

st.set_page_config(page_title="GeoQuery", page_icon="🛰️", layout="wide")


# ---------------------------------------------------------------------------
# Custom CSS for a premium, polished UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.8;
        font-size: 0.95rem;
    }

    /* Detection card styling */
    .detection-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        margin: 0.8rem 0;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #818cf8;
    }
    .metric-card .label {
        font-size: 0.75rem;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Timing badge */
    .timing-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* Example query chips */
    .query-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.8rem;
        cursor: pointer;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Better chat message styling */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading — cached so weights are loaded once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="🔄 Loading vision models (first run downloads ~4GB)...")
def load_models():
    from config import USE_YOLO
    if USE_YOLO:
        from models.yolo_detector import YOLODetector, HybridDetector
        detector = HybridDetector(YOLODetector())
    else:
        from models.detector import ObjectDetector
        detector = ObjectDetector()
    from models.captioner import ImageCaptioner
    from models.vqa import VisualQA
    return detector, ImageCaptioner(), VisualQA()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "image": None,
    "annotated_image": None,
    "detections": [],
    "caption": None,
    "chat_history": [],
    "timing": {},           # inference timing data
    "upload_processed": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛰️ GeoQuery")
    st.caption("Natural Language Understanding of RGB Imagery")

    st.divider()

    st.markdown("**Supported Classes**")
    class_icons = {
        "building": "🏢", "road": "🛤️", "vehicle": "🚗",
        "vegetation": "🌿", "water body": "💧", "open ground": "🏜️",
    }
    for c in TARGET_CLASSES:
        st.markdown(f"{class_icons.get(c, '•')} {c.capitalize()}")

    st.divider()

    # Device & system info
    device = get_device()
    device_label = {"cuda": "🟢 CUDA GPU", "mps": "🟡 Apple MPS", "cpu": "🔴 CPU"}.get(device, device)
    st.caption(f"**Device:** {device_label}")
    st.caption(f"**Max resolution:** {MAX_RESOLUTION}px")
    from config import USE_YOLO
    if USE_YOLO:
        st.caption("**Detector:** 🎯 YOLOv8 (fine-tuned)")
    else:
        st.caption("**Detector:** 🔍 Grounding DINO (zero-shot)")

    if st.session_state.timing:
        st.divider()
        st.markdown("**⏱️ Last Inference**")
        for label, secs in st.session_state.timing.items():
            st.caption(f"{label}: {secs:.1f}s")

    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True):
        for key, val in defaults.items():
            st.session_state[key] = val
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🛰️ GeoQuery</h1>
    <p>Upload an image, get an instant caption + object detections, then ask anything about it.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload an RGB image (.jpg / .png)",
    type=["jpg", "jpeg", "png"],
    help="Standard RGB images only. Max resolution 1024×1024px.",
)

if uploaded is not None and not st.session_state.upload_processed:
    err = validate_extension(uploaded.name)
    if err:
        st.error(err)
    else:
        image, load_err = load_and_prepare(uploaded.read())
        if load_err:
            st.error(load_err)
        else:
            detector, captioner, vqa = load_models()

            # Caption
            t0 = time.time()
            with st.spinner("✨ Generating caption..."):
                caption = captioner.caption(image)
            caption_time = time.time() - t0

            # Detection — use multi-pass for better accuracy
            t0 = time.time()
            with st.spinner("🔍 Detecting objects (multi-pass for accuracy)..."):
                detections = detector.detect_multipass(image)
            detect_time = time.time() - t0

            st.session_state.image = image
            st.session_state.caption = caption
            st.session_state.detections = detections
            st.session_state.annotated_image = draw_detections(image, detections)
            st.session_state.chat_history = []
            st.session_state.upload_processed = True
            st.session_state.timing = {
                "Caption": caption_time,
                "Detection": detect_time,
            }
            st.rerun()


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
if st.session_state.image is not None:
    col_img, col_chat = st.columns([1, 1], gap="large")

    with col_img:
        # Annotated image (clickable!)
        from streamlit_image_coordinates import streamlit_image_coordinates
        from utils.intent_router import Intent
        
        coords = streamlit_image_coordinates(
            st.session_state.annotated_image,
            key="image_clicker",
            use_column_width=True
        )
        
        if coords is not None and coords != st.session_state.get("last_coords"):
            st.session_state["last_coords"] = coords
            x, y = coords["x"], coords["y"]
            
            clicked_det = None
            # Reverse iterate to pick top-most box if overlapping
            for det in reversed(st.session_state.detections):
                bx1, by1, bx2, by2 = det.box
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    clicked_det = det
                    break
                    
            if clicked_det:
                q = f"Describe the {clicked_det.label} I clicked on."
                st.session_state.chat_history.append(("user", q))
                _, _, vqa = load_models()
                # _answer_attribute handles color and size
                ans = vqa._answer_attribute(
                    Intent("attribute", clicked_det.label, True, None),
                    [clicked_det],
                    st.session_state.image,
                    "describe it"
                )
                st.session_state.chat_history.append(("assistant", ans))
                st.rerun()

        # Caption
        st.markdown(f"**📝 Caption:** *{st.session_state.caption}*")

        # Timing badge
        timing = st.session_state.timing
        if timing:
            times = " • ".join(f"{k}: {v:.1f}s" for k, v in timing.items())
            st.markdown(f'<span class="timing-badge">⏱️ {times}</span>', unsafe_allow_html=True)

        # Detection metrics
        dets = st.session_state.detections
        from collections import Counter
        class_counts = Counter(d.label for d in dets)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="value">{len(dets)}</div>
                <div class="label">Objects Detected</div>
            </div>
            <div class="metric-card">
                <div class="value">{len(class_counts)}</div>
                <div class="label">Classes Found</div>
            </div>
            <div class="metric-card">
                <div class="value">{max((d.confidence for d in dets), default=0):.0%}</div>
                <div class="label">Top Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Detection summary
        with st.expander(f"📋 Detection Details ({len(dets)} objects)", expanded=True):
            if not dets:
                st.info("No objects from the supported classes were detected.")
            else:
                for det in sorted(dets, key=lambda d: -d.confidence):
                    crop = crop_box(st.session_state.image, det.box)
                    color = dominant_color_name(crop)
                    size = relative_size_label(det.box, st.session_state.image)
                    icon = class_icons.get(det.label, "•")
                    st.markdown(
                        f"{icon} **{det.label.capitalize()}** — "
                        f"`{det.confidence:.0%}` · {color} · {size}"
                    )

        # Per-class breakdown
        if class_counts:
            with st.expander("📊 Class Breakdown"):
                for cls in TARGET_CLASSES:
                    cnt = class_counts.get(cls, 0)
                    if cnt > 0:
                        icon = class_icons.get(cls, "•")
                        avg_conf = sum(d.confidence for d in dets if d.label == cls) / cnt
                        st.markdown(f"{icon} **{cls.capitalize()}**: {cnt} detected (avg conf: {avg_conf:.0%})")

        # PDF report download
        if st.session_state.chat_history:
            pdf_bytes = build_pdf_report(
                st.session_state.annotated_image,
                st.session_state.caption,
                st.session_state.chat_history,
                detections=st.session_state.detections,
                original_image=st.session_state.image,
            )
            st.download_button(
                "📄 Download Session Report (PDF)",
                data=pdf_bytes,
                file_name="geoquery_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # ------------------------------------------------------------------
    # Chat column
    # ------------------------------------------------------------------
    with col_chat:
        st.markdown("### 💬 Ask about this image")

        # Example queries
        if not st.session_state.chat_history:
            st.caption("Try one of these example queries:")
            example_queries = [
                "How many buildings are visible?",
                "Is there a road in the lower half?",
                "What colour is the largest building?",
                "What fraction of the image is vegetation?",
                "Count the vehicles in the image.",
                "Describe the open ground region.",
                "Mark all water bodies.",
            ]
            cols = st.columns(2)
            for i, eq in enumerate(example_queries):
                with cols[i % 2]:
                    if st.button(eq, key=f"eq_{i}", use_container_width=True):
                        st.session_state.chat_history.append(("user", eq))
                        st.rerun()

        # Chat display
        chat_container = st.container(height=450)
        with chat_container:
            for role, text in st.session_state.chat_history:
                with st.chat_message(role):
                    st.markdown(text)

        # Chat input
        query = st.chat_input("e.g. 'How many vehicles are there?'")

        # Process query (from text input or example button click)
        pending_query = None
        if query:
            pending_query = query
        elif (st.session_state.chat_history
              and len(st.session_state.chat_history) % 2 == 1
              and st.session_state.chat_history[-1][0] == "user"):
            # An example button was clicked — process the last user message
            pending_query = st.session_state.chat_history[-1][1]

        if pending_query:
            if pending_query == query:
                st.session_state.chat_history.append(("user", pending_query))

            detector, captioner, vqa = load_models()
            intent = classify(pending_query, st.session_state.chat_history)

            t0 = time.time()

            if intent.query_type == "detect":
                prompt = f"{intent.target_class}." if intent.target_class else None
                from config import DETECTION_PROMPT
                new_dets = detector.detect(
                    st.session_state.image,
                    prompt=prompt or DETECTION_PROMPT,
                )
                st.session_state.detections = new_dets
                st.session_state.annotated_image = draw_detections(st.session_state.image, new_dets)
                n = len(new_dets)
                label = intent.target_class or "objects"
                if n > 0:
                    confs = [d.confidence for d in new_dets]
                    response = (
                        f"Marked **{n} {label}{'s' if n > 1 else ''}** on the image "
                        f"(confidence range: {min(confs):.0%}–{max(confs):.0%}). "
                        f"See the updated image on the left."
                    )
                else:
                    response = f"I couldn't detect any {label} in this image."
            else:
                response = vqa.answer(st.session_state.image, pending_query, st.session_state.detections)

            elapsed = time.time() - t0
            st.session_state.timing["Last Query"] = elapsed

            st.session_state.chat_history.append(("assistant", response))
            st.rerun()

else:
    # ---------------------------------------------------------------------------
    # Welcome screen (no image uploaded yet)
    # ---------------------------------------------------------------------------
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📝 Captioning")
        st.markdown("Upload an image and get an automatic, accurate natural language caption describing the scene.")
    with col2:
        st.markdown("### 🔍 Detection")
        st.markdown("Detect and localize 6 object classes with bounding boxes, labels, and confidence scores.")
    with col3:
        st.markdown("### 💬 Visual Q&A")
        st.markdown("Ask natural language questions — counting, presence, attributes, spatial, and more.")

    st.markdown("---")
    st.markdown("#### 🎯 Supported Classes")
    cols = st.columns(6)
    for i, cls in enumerate(TARGET_CLASSES):
        with cols[i]:
            icon = class_icons.get(cls, "•")
            st.markdown(f"**{icon}**\n\n{cls.capitalize()}")

    st.markdown("---")
    st.caption("Built for the IIT Ropar Mock Inter IIT Tech Meet 15.0 • All models run locally • No commercial APIs")
