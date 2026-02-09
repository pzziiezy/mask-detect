import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
import threading
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

TF_AVAILABLE = False
TF_IMPORT_ERROR = ""
keras = None
RUNNING_ON_STREAMLIT_CLOUD = os.getenv("USER") == "appuser" and os.path.exists("/app")
ENABLE_TF_INFERENCE = os.getenv("ENABLE_TF_INFERENCE")
if ENABLE_TF_INFERENCE is None:
    ENABLE_TF_INFERENCE = not RUNNING_ON_STREAMLIT_CLOUD
else:
    ENABLE_TF_INFERENCE = ENABLE_TF_INFERENCE.lower() in {"1", "true", "yes", "on"}

if not ENABLE_TF_INFERENCE:
    TF_IMPORT_ERROR = (
        "TensorFlow inference is disabled for this runtime "
        "(set ENABLE_TF_INFERENCE=1 to force-enable)."
    )
elif sys.version_info >= (3, 13):
    TF_IMPORT_ERROR = (
        f"Python {sys.version_info.major}.{sys.version_info.minor} detected. "
        "TensorFlow-backed model inference is disabled on this Python version."
    )
else:
    try:
        # Use deterministic CPU path to avoid oneDNN-related startup instability.
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        from tensorflow import keras as tf_keras

        keras = tf_keras
        TF_AVAILABLE = True
    except Exception as exc:
        TF_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

WEBRTC_IMPORT_ERROR = ""
try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception as exc:
    WEBRTC_AVAILABLE = False
    WEBRTC_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

st.set_page_config(
    page_title="Mask Detection Dashboard",
    page_icon="😷",
    layout="wide",
)


def inject_styles():
    st.markdown(
        """
        <style>
            :root {
                --bg-main: #f0f4f8;
                --bg-accent: #e2e8f0;
                --text-main: #1a202c;
                --text-sub: #4a5568;
                --panel: #ffffff;
                --primary: #4361ee;
                --primary-light: #6b82f7;
                --success: #0ea575;
                --danger: #ef4444;
                --warning: #f59e0b;
                --border: #e2e8f0;
                --shadow-sm: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
                --shadow-md: 0 4px 14px rgba(67,97,238,.10), 0 2px 6px rgba(0,0,0,.04);
                --shadow-lg: 0 10px 28px rgba(67,97,238,.12), 0 4px 10px rgba(0,0,0,.04);
                --radius: 14px;
                color-scheme: light;
            }

            /* --- iOS safe-area & touch optimisation --- */
            html, body {
                -webkit-text-size-adjust: 100%;
                -webkit-tap-highlight-color: transparent;
                -webkit-overflow-scrolling: touch;
                overscroll-behavior-y: none;
            }

            .stApp {
                background:
                    radial-gradient(circle at 12% 8%, rgba(67,97,238,.07) 0%, transparent 40%),
                    radial-gradient(circle at 85% 15%, rgba(99,179,237,.08) 0%, transparent 45%),
                    radial-gradient(circle at 50% 90%, rgba(139,92,246,.05) 0%, transparent 35%),
                    linear-gradient(170deg, var(--bg-main) 0%, #dfe6f0 50%, var(--bg-accent) 100%);
                color: var(--text-main);
            }

            .main, .main .block-container {
                color: var(--text-main);
            }

            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 1.5rem;
            }

            /* ---------- Hero Banner ---------- */
            .hero {
                background: linear-gradient(135deg, #4361ee 0%, #3a56d4 40%, #7c3aed 100%);
                color: #ffffff;
                border-radius: var(--radius);
                padding: 1.3rem 1.4rem;
                margin-bottom: 1.1rem;
                border: 1px solid rgba(255,255,255,.18);
                box-shadow: var(--shadow-lg);
                position: relative;
                overflow: hidden;
            }

            .hero::before {
                content: "";
                position: absolute;
                top: -30%;
                right: -10%;
                width: 220px;
                height: 220px;
                background: radial-gradient(circle, rgba(255,255,255,.12) 0%, transparent 70%);
                border-radius: 50%;
                pointer-events: none;
            }

            .hero h1 {
                margin: 0 0 .3rem 0;
                font-size: 1.6rem;
                font-weight: 800;
                letter-spacing: .3px;
                text-shadow: 0 1px 2px rgba(0,0,0,.12);
            }

            .hero p {
                margin: 0;
                opacity: .92;
                font-size: .95rem;
                font-weight: 400;
            }

            /* ---------- Sidebar ---------- */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            }

            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] span {
                color: #e2e8f0 !important;
            }

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] b {
                color: #93c5fd !important;
            }

            .side-card {
                background: rgba(255,255,255,.07);
                border: 1px solid rgba(255,255,255,.14);
                border-radius: 12px;
                padding: .85rem 1rem;
                margin-top: .7rem;
                font-size: .88rem;
                line-height: 1.5;
                backdrop-filter: blur(6px);
                -webkit-backdrop-filter: blur(6px);
                color: #cbd5e1;
            }

            /* ---------- Section Titles ---------- */
            .section-title {
                margin: .2rem 0 .4rem 0;
                font-size: 1.08rem;
                font-weight: 700;
                color: var(--text-main);
            }

            /* ---------- Radio Buttons ---------- */
            .main [data-testid="stRadio"] div[role="radiogroup"] {
                gap: .75rem;
            }

            /* ---------- Metric Cards ---------- */
            [data-testid="stMetric"] {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: .6rem .75rem;
                box-shadow: var(--shadow-sm);
                transition: box-shadow .2s ease, transform .2s ease;
            }

            [data-testid="stMetric"]:hover {
                box-shadow: var(--shadow-md);
                transform: translateY(-1px);
            }

            [data-testid="stMetricValue"] {
                color: var(--primary) !important;
                font-weight: 700;
            }

            /* ---------- Buttons ---------- */
            .stButton > button {
                border-radius: 10px;
                font-weight: 600;
                transition: all .2s ease;
            }

            .stButton > button:active {
                transform: scale(.97);
            }

            /* ---------- Expander ---------- */
            [data-testid="stExpander"] {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                box-shadow: var(--shadow-sm);
            }

            /* ---------- Progress Bar ---------- */
            .stProgress > div > div {
                border-radius: 8px;
            }

            /* ---------- Plotly Charts ---------- */
            [data-testid="stPlotlyChart"] {
                border-radius: var(--radius);
                overflow: hidden;
                box-shadow: var(--shadow-sm);
            }

            /* ========== Mobile (max-width: 768px) ========== */
            @media (max-width: 768px) {
                .block-container {
                    padding-top: .8rem;
                    padding-left: .65rem;
                    padding-right: .65rem;
                    padding-bottom: calc(.8rem + env(safe-area-inset-bottom, 0px));
                }

                .hero {
                    border-radius: 12px;
                    padding: 1rem 1.05rem;
                }

                .hero::before { display: none; }

                .hero h1 {
                    font-size: 1.3rem;
                    line-height: 1.25;
                }

                .hero p {
                    font-size: .88rem;
                }

                .main [data-testid="stRadio"] div[role="radiogroup"] {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: .45rem;
                }

                [data-testid="stMetric"] {
                    padding: .45rem .55rem;
                    border-radius: 10px;
                }

                /* Larger touch targets on mobile */
                .stButton > button,
                [data-testid="stRadio"] label {
                    min-height: 44px;
                }

                /* Camera / file-upload input full-width */
                [data-testid="stFileUploader"],
                [data-testid="stCameraInput"] {
                    width: 100% !important;
                }

                [data-testid="stCameraInput"] video,
                [data-testid="stCameraInput"] img {
                    border-radius: 10px;
                    max-height: 55vh;
                    object-fit: contain;
                }
            }

            /* ========== Small phones (max-width: 480px) ========== */
            @media (max-width: 480px) {
                .hero h1 { font-size: 1.15rem; }
                .hero p  { font-size: .82rem; }

                [data-testid="stMetric"] {
                    padding: .35rem .45rem;
                }

                .section-title { font-size: .98rem; }
            }

            /* iOS standalone (Add to Home Screen) */
            @media (display-mode: standalone) {
                .block-container {
                    padding-top: calc(.8rem + env(safe-area-inset-top, 0px));
                }
            }
        </style>
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        """,
        unsafe_allow_html=True,
    )


def play_sound(sound_type):
    """Render browser alert sound."""
    if sound_type == "success":
        st.markdown(
            """
            <audio autoplay>
                <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2i779+eUBELTKXh8bllHAU2kNbz0HwqBSl+zPLaizsKGGS56+mmWBQLTKHd8b1pIgUqf8rx3I4+CRxptO/bm1gTC0yi4fG9aiIFK4DN8tyJOQcad7zs4J5SEQxPpuHxtWYdBjiP1vLReisGKn/M8dqLOwoYZLnr6aZYFAtMod3xvWkiBSp/yvHcjj4JHGm079ubWBMLTKLh8b1qIgUrfw==" type="audio/wav">
            </audio>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <audio autoplay>
                <source src="data:audio/wav;base64,UklGRi4HAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoHAACAhpCQkJCQkJCQjo2MioiEgX1+f4KFioyOj5CQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkI+OjIqIhIF9fn+ChYqMjo+QkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCPjoyKiISBfX5/goWKjI6PkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQj46MioiEgX1+f4KFioyOj5CQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkI+OjIqIhIF9fn+ChYqMjo+QkJCQkA==" type="audio/wav">
            </audio>
            """,
            unsafe_allow_html=True,
        )


import tflite_runtime.interpreter as tflite

@st.cache_resource
def load_model():
    try:
        # โหลดโมเดลผ่าน TFLite Interpreter
        interpreter = tflite.Interpreter(model_path="models/mask_detector.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def detect_mask(image, threshold=0.3):
    img_size = 128
    labels = ["No Mask", "With Mask"]
    colors = [(255, 0, 0), (0, 255, 0)]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []
    for (x, y, w, h) in faces:
        face_img = image[y : y + h, x : x + w]
        face_img = cv2.resize(face_img, (img_size, img_size))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        prediction = model.predict(face_img, verbose=0)[0][0]
        label_idx = 1 if prediction > threshold else 0
        label = labels[label_idx]
        confidence = prediction if label_idx == 1 else (1 - prediction)
        color = colors[label_idx]

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        text = f"{label}: {confidence*100:.1f}%"

        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (x, y - 35), (x + text_width + 10, y), color, -1)
        cv2.putText(
            image,
            text,
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        results.append(
            {
                "label": label,
                "confidence": confidence,
                "has_mask": label_idx == 1,
            }
        )

    return image, results, len(faces)


if WEBRTC_AVAILABLE:
    class MaskVideoProcessor:
        def __init__(self):
            self.lock = threading.Lock()
            self.latest_stats = {
                "num_faces": 0,
                "with_mask": 0,
                "without_mask": 0,
                "error": "",
            }

        def recv(self, frame):
            try:
                image = frame.to_ndarray(format="bgr24")
                threshold = st.session_state.get("threshold", 0.3)
                result_frame, results, num_faces = detect_mask(image, threshold)
                with_mask = sum(1 for r in results if r["has_mask"])
                without_mask = num_faces - with_mask

                with self.lock:
                    self.latest_stats = {
                        "num_faces": num_faces,
                        "with_mask": with_mask,
                        "without_mask": without_mask,
                        "error": "",
                    }

                return av.VideoFrame.from_ndarray(result_frame, format="bgr24")
            except Exception as exc:
                with self.lock:
                    self.latest_stats = {
                        "num_faces": 0,
                        "with_mask": 0,
                        "without_mask": 0,
                        "error": str(exc),
                    }
                return frame

        def get_stats(self):
            with self.lock:
                return dict(self.latest_stats)


inject_styles()

if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

if "webcam_last_sound_time" not in st.session_state:
    st.session_state.webcam_last_sound_time = 0.0

if "webcam_last_history_time" not in st.session_state:
    st.session_state.webcam_last_history_time = 0.0

model = load_model()
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if model is None:
    st.error(
        "Model initialization failed. "
        f"{TF_IMPORT_ERROR or 'TensorFlow is unavailable.'} "
        "On Streamlit Community Cloud, deploy with Python 3.11 or 3.12 from "
        "Advanced settings, then redeploy."
    )
    st.stop()

PAGE_DETECTION = "Detection"
PAGE_REPORTS = "Reports"
MODE_UPLOAD = "Upload Image"
MODE_WEBCAM = "Webcam (Real-time)"

st.sidebar.markdown("## Control Panel")
page = st.sidebar.radio("Navigate", [PAGE_DETECTION, PAGE_REPORTS])
st.sidebar.markdown("---")
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.3, 0.05)
st.session_state.threshold = threshold
st.sidebar.markdown(
    """
    <div class="side-card">
        <b>Model Summary</b><br>
        Accuracy: 99.80%<br>
        Dataset: 12K Images<br>
        Audio Alerts: ON
    </div>
    """,
    unsafe_allow_html=True,
)

if page == PAGE_DETECTION:
    st.markdown(
        """
        <div class="hero">
            <h1>Face Mask Detection</h1>
            <p>Analyze uploaded images or webcam feed in real-time with compliance statistics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("How to use", expanded=False):
        st.markdown(
            """
            1. Select `Upload Image` or `Webcam (Real-time)`.
            2. Adjust `Detection Threshold` in the left panel.
            3. System displays detections and logs results to reports automatically.
            """
        )

    detection_method = st.radio(
        "Detection Method",
        [MODE_UPLOAD, MODE_WEBCAM],
        horizontal=True,
    )

    if detection_method == MODE_UPLOAD:
        st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
        st.caption("Supported: JPG, JPEG, PNG")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Original")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

            with st.spinner("Detecting..."):
                result_image, results, num_faces = detect_mask(image.copy(), threshold)

            with col2:
                st.caption("Detected")
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)

            without_mask = sum(1 for r in results if not r["has_mask"])
            if without_mask > 0:
                play_sound("warning")
            elif num_faces > 0:
                play_sound("success")

            with_mask = sum(1 for r in results if r["has_mask"])
            st.session_state.detection_history.append(
                {
                    "timestamp": datetime.now(),
                    "total_faces": num_faces,
                    "with_mask": with_mask,
                    "without_mask": without_mask,
                    "method": "Upload Image",
                }
            )

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Faces", num_faces)
            with m2:
                st.metric("With Mask", with_mask)
            with m3:
                st.metric("Without Mask", without_mask)
            with m4:
                compliance = (with_mask / num_faces) * 100 if num_faces > 0 else 0
                st.metric("Compliance", f"{compliance:.1f}%")

            st.progress(compliance / 100 if num_faces > 0 else 0.0, text="Compliance Rate")
            if without_mask > 0:
                st.error("Warning: People without masks detected.")
            elif num_faces > 0:
                st.success("All detected people are wearing masks.")

    else:
        st.markdown(
            '<div class="section-title">Real-time Webcam Detection</div>',
            unsafe_allow_html=True,
        )
        if WEBRTC_AVAILABLE:
            st.caption("Use `Live Stream` first. If camera fails on mobile/browser, switch to `Capture` mode.")
            camera_mode_options = ["Live Stream (WebRTC)", "Capture (Mobile Safe)"]
        else:
            st.caption("WebRTC is unavailable in this deployment. Use `Capture (Mobile Safe)` now.")
            camera_mode_options = ["Capture (Mobile Safe)"]

        camera_mode = st.radio("Camera Mode", camera_mode_options, horizontal=True)

        if camera_mode == "Live Stream (WebRTC)":
            if not WEBRTC_AVAILABLE:
                st.error("WebRTC dependency is missing. Use `Capture (Mobile Safe)` now and redeploy.")
                if WEBRTC_IMPORT_ERROR:
                    st.caption(f"Import detail: {WEBRTC_IMPORT_ERROR}")
            else:
                webrtc_ctx = webrtc_streamer(
                    key="mask-detection-webrtc",
                    mode=WebRtcMode.SENDRECV,
                    media_stream_constraints={
                        "video": {"facingMode": "user", "width": {"ideal": 640}, "height": {"ideal": 480}},
                        "audio": False,
                    },
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    video_html_attrs={
                        "autoPlay": True,
                        "playsInline": True,
                        "muted": True,
                    },
                    video_processor_factory=MaskVideoProcessor,
                    async_processing=True,
                )

                stats_placeholder = st.empty()
                alert_placeholder = st.empty()
                stats = {
                    "num_faces": 0,
                    "with_mask": 0,
                    "without_mask": 0,
                    "error": "",
                }

                if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                    stats = webrtc_ctx.video_processor.get_stats()

                num_faces = stats["num_faces"]
                with_mask = stats["with_mask"]
                without_mask = stats["without_mask"]
                compliance = (with_mask / num_faces * 100) if num_faces > 0 else 0

                with stats_placeholder.container():
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Faces", num_faces)
                    with c2:
                        st.metric("With Mask", with_mask)
                    with c3:
                        st.metric("Without Mask", without_mask)
                    with c4:
                        st.metric("Compliance", f"{compliance:.0f}%")

                if num_faces > 0:
                    current_time = time.time()
                    if current_time - st.session_state.webcam_last_history_time >= 3:
                        st.session_state.detection_history.append(
                            {
                                "timestamp": datetime.now(),
                                "total_faces": num_faces,
                                "with_mask": with_mask,
                                "without_mask": without_mask,
                                "method": "Webcam",
                            }
                        )
                        st.session_state.webcam_last_history_time = current_time

                with alert_placeholder:
                    if stats["error"]:
                        st.warning(f"Frame processing error: {stats['error']}")
                    elif not webrtc_ctx.state.playing:
                        st.info("Press `START` to activate webcam.")
                    elif num_faces == 0:
                        st.info("No face detected.")
                    elif without_mask > 0:
                        st.error("Warning: No mask detected.")
                    else:
                        st.success("All wearing masks.")

                st.caption(
                    "If webcam does not start on mobile: allow browser camera permission, use HTTPS URL, "
                    "or switch to `Capture (Mobile Safe)`."
                )
        else:
            captured_file = st.camera_input("Capture from mobile/desktop camera")
            if captured_file is not None:
                file_bytes = np.asarray(bytearray(captured_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Captured")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

                result_image, results, num_faces = detect_mask(image.copy(), threshold)
                with_mask = sum(1 for r in results if r["has_mask"])
                without_mask = num_faces - with_mask
                compliance = (with_mask / num_faces * 100) if num_faces > 0 else 0

                with col2:
                    st.caption("Detected")
                    st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.session_state.detection_history.append(
                    {
                        "timestamp": datetime.now(),
                        "total_faces": num_faces,
                        "with_mask": with_mask,
                        "without_mask": without_mask,
                        "method": "Camera Capture",
                    }
                )

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Faces", num_faces)
                with m2:
                    st.metric("With Mask", with_mask)
                with m3:
                    st.metric("Without Mask", without_mask)
                with m4:
                    st.metric("Compliance", f"{compliance:.1f}%")

else:
    st.markdown(
        """
        <div class="hero">
            <h1>Detection Reports</h1>
            <p>Review detection history, compliance trends, and mask usage distribution.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(st.session_state.detection_history) == 0:
        st.info("No detection data yet. Start detection to generate reports.")
    else:
        df = pd.DataFrame(st.session_state.detection_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        total_detections = len(df)
        total_faces = int(df["total_faces"].sum())
        total_with_mask = int(df["with_mask"].sum())
        total_without_mask = int(df["without_mask"].sum())

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Detections", total_detections)
        with k2:
            st.metric("Total Faces", total_faces)
        with k3:
            st.metric("With Mask", total_with_mask)
        with k4:
            st.metric("Without Mask", total_without_mask)

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">Compliance Over Time</div>', unsafe_allow_html=True)
            df["compliance"] = (df["with_mask"] / df["total_faces"] * 100).fillna(0)
            fig = px.line(
                df,
                x="timestamp",
                y="compliance",
                labels={"compliance": "Compliance (%)", "timestamp": "Time"},
            )
            fig.update_traces(line_color="#4361ee", line_width=3)
            fig.update_layout(
                margin=dict(l=10, r=10, t=20, b=0),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font_color="#1a202c",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-title">Overall Distribution</div>', unsafe_allow_html=True)
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["With Mask", "Without Mask"],
                        values=[total_with_mask, total_without_mask],
                        marker_colors=["#4361ee", "#ef4444"],
                        hole=0.45,
                    )
                ]
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Detection Breakdown</div>', unsafe_allow_html=True)

        df_melted = df[["timestamp", "with_mask", "without_mask"]].melt(
            id_vars="timestamp",
            value_vars=["with_mask", "without_mask"],
            var_name="Status",
            value_name="Count",
        )
        df_melted["Status"] = df_melted["Status"].map(
            {"with_mask": "With Mask", "without_mask": "Without Mask"}
        )

        fig = px.bar(
            df_melted,
            x="timestamp",
            y="Count",
            color="Status",
            color_discrete_map={"With Mask": "#4361ee", "Without Mask": "#ef4444"},
            barmode="group",
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=0),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font_color="#1a202c",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Detection History</div>', unsafe_allow_html=True)
        df_display = df.copy()
        df_display["timestamp"] = df_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df_display = df_display.rename(
            columns={
                "timestamp": "Time",
                "total_faces": "Total Faces",
                "with_mask": "With Mask",
                "without_mask": "Without Mask",
                "method": "Method",
            }
        )
        st.dataframe(df_display, use_container_width=True, height=380)

        st.markdown("---")
        if st.button("Clear All History", type="secondary"):
            st.session_state.detection_history = []
            st.rerun()
