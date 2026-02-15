from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import numpy as np
from tensorflow import keras
import base64
from io import BytesIO
from PIL import Image
import json
from datetime import datetime
from collections import deque
import os
import requests

# ดาวน์โหลดโมเดลจาก Google Drive
def download_model_from_gdrive(file_id, destination):
    if os.path.exists(destination):
        print(f"✓ Model already exists: {destination}")
        return
    
    print(f"Downloading model from Google Drive...")
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    session = requests.Session()
    response = session.get(URL, stream=True)
    
    # บันทึกไฟล์
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    
    print(f"✓ Model downloaded: {destination}")

# ใส่ Google Drive File ID ของคุณที่นี่
GDRIVE_FILE_ID = "1wKuD0D_Bz_lv8Mryyt37-r3Aph33aK5l"
MODEL_PATH = 'models/mask_detector_production.h5'

# ดาวน์โหลดก่อนโหลดโมเดล
download_model_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)


app = Flask(__name__)

# โหลดโมเดล
ACTIVE_MODEL_PATH = None
try:
    # ลำดับความสำคัญ: โมเดลใหม่ → VGG → CNN
    new_model_path = 'models/mask_detector_production.h5'
    vgg_model_path = 'models/mask_detector_vgg16.h5'
    cnn_model_path = 'models/mask_detector.h5'

    if os.path.exists(new_model_path):
        ACTIVE_MODEL_PATH = new_model_path
        model = keras.models.load_model(new_model_path)
        IMG_SIZE = 224
        print(f"✓ Loaded NEW model: {new_model_path}")
    elif os.path.exists(vgg_model_path):
        ACTIVE_MODEL_PATH = vgg_model_path
        model = keras.models.load_model(vgg_model_path)
        IMG_SIZE = 224
        print(f"✓ Loaded VGG model: {vgg_model_path}")
    else:
        ACTIVE_MODEL_PATH = cnn_model_path
        model = keras.models.load_model(cnn_model_path)
        IMG_SIZE = 128
        print(f"✓ Loaded CNN model: {cnn_model_path}")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    if face_cascade.empty():
        raise RuntimeError("Cannot load frontal face cascade")
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False

DEFAULT_MODEL_METADATA = {
    'class_indices': {'with_mask': 0, 'without_mask': 1},
    'positive_label': 'without_mask',
    'mask_threshold': 0.5,
    'img_size': 224
}


def load_model_metadata():
    metadata_path = 'models/model_metadata.json'
    metadata = dict(DEFAULT_MODEL_METADATA)
    if not os.path.exists(metadata_path):
        return metadata

    try:
        with open(metadata_path, 'r', encoding='utf-8') as fp:
            loaded = json.load(fp)
        if isinstance(loaded, dict):
            metadata.update(loaded)
    except Exception as err:
        print(f"Metadata load warning: {err}")
    return metadata


MODEL_METADATA = load_model_metadata()
MASK_THRESHOLD = float(MODEL_METADATA.get('mask_threshold', 0.5))

# เก็บสถิติ
stats = {
    'total_scans': 0,
    'with_mask': 0,
    'without_mask': 0,
    'last_detection': None
}

# Temporal smoothing - เก็บ predictions ล่าสุด
prediction_history = deque(maxlen=3)

def detect_mask_advanced_legacy(image, threshold=0.65, min_confidence=0.75):
    """
    Advanced detection with quality checks
    """
    labels = ['ไม่สวมแมสก์', 'สวมแมสก์']
    draw_labels = ['No Mask', 'With Mask']
    colors = [(0, 0, 255), (0, 255, 0)]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with stricter parameters
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,  # ลดจาก 1.3 เพื่อความแม่นยำ
        minNeighbors=5,   # เพิ่มเพื่อลด false positive
        minSize=(60, 60)  # ใบหน้าต้องใหญ่พอ
    )
    
    results = []
    
    for (x, y, w, h) in faces:
        try:
            # Expand ROI slightly
            padding = 10
            y1 = max(0, y - padding)
            y2 = min(image.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(image.shape[1], x + w + padding)
            
            face_img = image[y1:y2, x1:x2]
            
            if face_img.size == 0 or face_img.shape[0] < 30 or face_img.shape[1] < 30:
                continue
            
            # Quality check - ตรวจสอบความคมชัด
            laplacian_var = cv2.Laplacian(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if laplacian_var < 50:  # ภาพเบลอเกินไป
                continue
            
            # Resize and normalize
            face_img_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            face_img_normalized = face_img_resized.astype('float32') / 255.0
            face_img_batch = np.expand_dims(face_img_normalized, axis=0)
            
            # Predict
            prediction = model.predict(face_img_batch, verbose=0)[0][0]
            
            # Temporal smoothing - average last 3 predictions
            prediction_history.append(prediction)
            smoothed_prediction = np.mean(list(prediction_history))
            
            # Apply threshold with higher confidence requirement
            if smoothed_prediction > threshold:
                label_idx = 1
                confidence = smoothed_prediction
            else:
                label_idx = 0
                confidence = 1 - smoothed_prediction
            
            # Skip low confidence detections
            if confidence < min_confidence:
                continue
            
            label = labels[label_idx]
            draw_label = draw_labels[label_idx]
            color = colors[label_idx]
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
            
            # Draw label with background
            text = f"{draw_label}: {confidence*100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Background rectangle
            cv2.rectangle(image, 
                         (x, y - text_height - 15), 
                         (x + text_width + 10, y), 
                         color, -1)
            
            # Text
            cv2.putText(image, text, (x + 5, y - 10), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Quality indicator
            quality_text = f"Quality: {int(laplacian_var)}"
            cv2.putText(image, quality_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            results.append({
                'label': label,
                'confidence': float(confidence),
                'has_mask': label_idx == 1,
                'quality_score': float(laplacian_var),
                'bbox': {
                    'x': int(x),
                    'y': int(y),
                    'w': int(w),
                    'h': int(h)
                }
            })
            
        except Exception as e:
            print(f"Face processing error: {e}")
            continue
    
    return image, results


def enhance_low_light(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # เพิ่ม clipLimit
    l_channel = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return cv2.convertScaleAbs(enhanced_bgr, alpha=1.2, beta=10)  # เพิ่มความสว่าง


def iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (aw * ah) + (bw * bh) - inter
    return (inter / union) if union > 0 else 0.0


def deduplicate_faces(boxes, iou_threshold=0.30):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for candidate in boxes:
        if all(iou(candidate, k) < iou_threshold for k in kept):
            kept.append(candidate)
    return kept


def detect_faces_robust(image):
    h, w = image.shape[:2]
    enhanced = enhance_low_light(image)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    all_faces = []
    angles = [0, -10, 10, -20, 20]
    center = (w // 2, h // 2)

    for angle in angles:
        if angle == 0:
            rotated = gray
            inv_m = None
        else:
            m = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                gray, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            inv_m = cv2.invertAffineTransform(m)

        frontal = face_cascade.detectMultiScale(rotated, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))
        profiles = []
        if profile_face_cascade is not None and not profile_face_cascade.empty():
            profiles = profile_face_cascade.detectMultiScale(
                rotated, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50)
            )

        for source in (frontal, profiles):
            for (x, y, fw, fh) in source:
                if inv_m is None:
                    bx, by, bw, bh = int(x), int(y), int(fw), int(fh)
                else:
                    corners = np.array(
                        [[[x, y], [x + fw, y], [x, y + fh], [x + fw, y + fh]]], dtype=np.float32
                    )
                    mapped = cv2.transform(corners, inv_m)[0]
                    min_x = int(max(0, np.min(mapped[:, 0])))
                    min_y = int(max(0, np.min(mapped[:, 1])))
                    max_x = int(min(w - 1, np.max(mapped[:, 0])))
                    max_y = int(min(h - 1, np.max(mapped[:, 1])))
                    bx, by, bw, bh = min_x, min_y, max_x - min_x, max_y - min_y

                if bw >= 35 and bh >= 35:
                    all_faces.append((bx, by, bw, bh))

    return deduplicate_faces(all_faces)


def predict_mask_score(face_img):
    resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
    flipped = cv2.flip(resized, 1)
    batch = np.stack([resized, flipped], axis=0)
    preds = model.predict(batch, verbose=0).flatten()
    return float(np.mean(preds))


def score_to_probabilities(score):
    positive_label = MODEL_METADATA.get('positive_label', 'without_mask')
    if positive_label == 'without_mask':
        prob_without_mask = float(score)
        prob_with_mask = 1.0 - prob_without_mask
    else:
        prob_with_mask = float(score)
        prob_without_mask = 1.0 - prob_with_mask
    return prob_with_mask, prob_without_mask


def detect_mask_advanced(image, threshold=None, min_confidence=0.60):
    labels = ['ไม่สวมแมสก์', 'สวมแมสก์']
    draw_labels = ['No Mask', 'With Mask']
    colors = [(0, 0, 255), (0, 255, 0)]
    results = []
    active_threshold = MASK_THRESHOLD if threshold is None else float(threshold)

    faces = detect_faces_robust(image)

    for (x, y, w, h) in faces:
        try:
            padding = 12
            y1 = max(0, y - padding)
            y2 = min(image.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(image.shape[1], x + w + padding)
            face_img = image[y1:y2, x1:x2]

            if face_img.size == 0 or face_img.shape[0] < 30 or face_img.shape[1] < 30:
                continue

            enhanced_face = enhance_low_light(face_img)
            laplacian_var = cv2.Laplacian(
                cv2.cvtColor(enhanced_face, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()

            prediction = predict_mask_score(enhanced_face)
            prob_with_mask, prob_without_mask = score_to_probabilities(prediction)
            has_mask = prob_with_mask >= active_threshold
            label_idx = 1 if has_mask else 0
            confidence = prob_with_mask if has_mask else prob_without_mask

            if laplacian_var < 20:
                confidence *= 0.9
            if confidence < min_confidence:
                continue

            label = labels[label_idx]
            draw_label = draw_labels[label_idx]
            color = colors[label_idx]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            text = f"{draw_label}: {confidence*100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(image, (x, y - text_height - 15), (x + text_width + 10, y), color, -1)
            cv2.putText(image, text, (x + 5, y - 10), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(
                image,
                f"Quality: {int(laplacian_var)}",
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            results.append({
                'label': label,
                'confidence': float(confidence),
                'has_mask': label_idx == 1,
                'quality_score': float(laplacian_var),
                'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            })

        except Exception as e:
            print(f"Face processing error: {e}")
            continue

    return image, results


@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Mask Detection System | Enterprise Solution</title>
        
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            :root {
                --primary: #6366f1;
                --primary-dark: #4f46e5;
                --success: #10b981;
                --danger: #ef4444;
                --warning: #f59e0b;
                --bg-dark: #0f172a;
                --bg-card: #1e293b;
                --text-primary: #f1f5f9;
                --text-secondary: #94a3b8;
                --border: #334155;
            }
            
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                color: var(--text-primary);
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            
            .header {
                background: rgba(15, 23, 42, 0.8);
                backdrop-filter: blur(10px);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 24px 32px;
                margin-bottom: 24px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }
            
            .logo {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .logo i {
                font-size: 32px;
                color: var(--primary);
            }
            
            .logo h1 {
                font-size: 24px;
                font-weight: 700;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .status-badge {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 8px 16px;
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid var(--success);
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
                color: var(--success);
            }
            
            .pulse {
                width: 8px;
                height: 8px;
                background: var(--success);
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .tabs {
                display: flex;
                gap: 12px;
                margin-bottom: 24px;
            }
            
            .tab {
                flex: 1;
                padding: 16px;
                background: rgba(30, 41, 59, 0.8);
                border: 1px solid var(--border);
                border-radius: 12px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                font-weight: 600;
            }
            
            .tab:hover {
                background: rgba(99, 102, 241, 0.2);
                border-color: var(--primary);
            }
            
            .tab.active {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                border-color: var(--primary);
            }
            
            .tab-content {
                display: none;
            }
            
            .tab-content.active {
                display: block;
            }
            
            .main-grid {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 24px;
                margin-bottom: 24px;
            }
            
            @media (max-width: 1024px) {
                .main-grid {
                    grid-template-columns: 1fr;
                }
            }
            
            .card {
                background: rgba(30, 41, 59, 0.8);
                backdrop-filter: blur(10px);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 24px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }
            
            .card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            
            .card-title {
                font-size: 18px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .card-title i {
                color: var(--primary);
            }
            
            .video-wrapper {
                position: relative;
                border-radius: 12px;
                overflow: hidden;
                background: #000;
                aspect-ratio: 16/9;
            }
            
            video, #resultCanvas, #overlayCanvas {
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }
            
            #resultCanvas {
                display: none;
            }

            #overlayCanvas {
                position: absolute;
                inset: 0;
                pointer-events: none;
                z-index: 2;
            }
            
            .video-overlay {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(to bottom, rgba(0,0,0,0.6) 0%, transparent 30%, transparent 70%, rgba(0,0,0,0.6) 100%);
                pointer-events: none;
                z-index: 1;
            }
            
            .recording-indicator {
                position: absolute;
                top: 16px;
                right: 16px;
                display: none;
                align-items: center;
                gap: 8px;
                padding: 8px 16px;
                background: rgba(239, 68, 68, 0.9);
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
                z-index: 3;
            }
            
            .recording-indicator.active {
                display: flex;
            }
            
            .rec-dot {
                width: 8px;
                height: 8px;
                background: white;
                border-radius: 50%;
                animation: blink 1s infinite;
            }
            
            @keyframes blink {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.3; }
            }
            
            .controls {
                display: flex;
                gap: 12px;
                margin-top: 16px;
            }
            
            .btn {
                flex: 1;
                padding: 14px 24px;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                color: white;
                box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 25px rgba(99, 102, 241, 0.4);
            }
            
            .btn-danger {
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                color: white;
                box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3);
            }
            
            .btn-danger:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 25px rgba(239, 68, 68, 0.4);
            }
            
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none !important;
            }
            
            .upload-area {
                border: 2px dashed var(--border);
                border-radius: 12px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                margin-bottom: 20px;
            }
            
            .upload-area:hover {
                border-color: var(--primary);
                background: rgba(99, 102, 241, 0.1);
            }
            
            .upload-area.dragover {
                border-color: var(--primary);
                background: rgba(99, 102, 241, 0.2);
            }
            
            .upload-area i {
                font-size: 48px;
                color: var(--primary);
                margin-bottom: 16px;
            }
            
            #fileInput {
                display: none;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 16px;
                margin-bottom: 20px;
            }
            
            .stat-card {
                background: rgba(15, 23, 42, 0.5);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
            }
            
            .stat-value {
                font-size: 32px;
                font-weight: 700;
                margin-bottom: 4px;
            }
            
            .stat-label {
                font-size: 13px;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .stat-card.success .stat-value {
                color: var(--success);
            }
            
            .stat-card.danger .stat-value {
                color: var(--danger);
            }
            
            .detection-list {
                max-height: 400px;
                overflow-y: auto;
            }
            
            .detection-item {
                background: rgba(15, 23, 42, 0.5);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 16px;
                margin-bottom: 12px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .detection-info {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .detection-icon {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
            }
            
            .detection-icon.success {
                background: rgba(16, 185, 129, 0.2);
                color: var(--success);
            }
            
            .detection-icon.danger {
                background: rgba(239, 68, 68, 0.2);
                color: var(--danger);
            }
            
            .detection-label {
                font-weight: 600;
                margin-bottom: 4px;
            }
            
            .detection-time {
                font-size: 12px;
                color: var(--text-secondary);
            }
            
            .confidence-badge {
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
                background: rgba(99, 102, 241, 0.2);
                color: var(--primary);
            }
            
            .detection-list::-webkit-scrollbar {
                width: 6px;
            }
            
            .detection-list::-webkit-scrollbar-track {
                background: rgba(15, 23, 42, 0.5);
                border-radius: 10px;
            }
            
            .detection-list::-webkit-scrollbar-thumb {
                background: var(--primary);
                border-radius: 10px;
            }
            
            .footer {
                text-align: center;
                padding: 20px;
                color: var(--text-secondary);
                font-size: 14px;
            }
            
            .loading {
                text-align: center;
                padding: 40px;
                color: var(--text-secondary);
            }
            
            .spinner {
                width: 40px;
                height: 40px;
                border: 4px solid rgba(99, 102, 241, 0.2);
                border-top-color: var(--primary);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 16px;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">
                    <i class="fas fa-shield-virus"></i>
                    <h1>AI Mask Detection System</h1>
                </div>
                <div class="status-badge">
                    <div class="pulse"></div>
                    <span>System Active</span>
                </div>
            </div>
            
            <div class="tabs">
                <div class="tab active" onclick="switchTab('realtime')">
                    <i class="fas fa-video"></i> Real-time Detection
                </div>
                <div class="tab" onclick="switchTab('upload')">
                    <i class="fas fa-upload"></i> Upload Image
                </div>
            </div>
            
            <!-- Real-time Tab -->
            <div id="realtime-content" class="tab-content active">
                <div class="main-grid">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title">
                                <i class="fas fa-video"></i>
                                Live Detection Feed
                            </div>
                        </div>
                        
                        <div class="video-wrapper">
                            <video id="video" autoplay playsinline></video>
                            <canvas id="overlayCanvas"></canvas>
                            <div class="video-overlay"></div>
                            <div class="recording-indicator" id="recordingIndicator">
                                <div class="rec-dot"></div>
                                <span>LIVE</span>
                            </div>
                        </div>
                        
                        <canvas id="canvas" style="display: none;"></canvas>
                        
                        <div class="controls">
                            <button class="btn btn-primary" id="startBtn" onclick="startCamera()">
                                <i class="fas fa-play"></i>
                                Start Detection
                            </button>
                            <button class="btn btn-danger" id="stopBtn" onclick="stopCamera()" disabled>
                                <i class="fas fa-stop"></i>
                                Stop Detection
                            </button>
                        </div>
                    </div>
                    
                    <div>
                        <div class="card" style="margin-bottom: 24px;">
                            <div class="card-header">
                                <div class="card-title">
                                    <i class="fas fa-chart-line"></i>
                                    Real-time Statistics
                                </div>
                            </div>
                            
                            <div class="stats-grid">
                                <div class="stat-card">
                                    <div class="stat-value" id="totalScans">0</div>
                                    <div class="stat-label">Total Scans</div>
                                </div>
                                <div class="stat-card success">
                                    <div class="stat-value" id="withMask">0</div>
                                    <div class="stat-label">With Mask</div>
                                </div>
                                <div class="stat-card danger">
                                    <div class="stat-value" id="withoutMask">0</div>
                                    <div class="stat-label">Without Mask</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-value" id="compliance">100%</div>
                                    <div class="stat-label">Compliance</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">
                                <div class="card-title">
                                    <i class="fas fa-list"></i>
                                    Detection History
                                </div>
                            </div>
                            
                            <div class="detection-list" id="detectionList">
                                <div class="loading">
                                    <div class="spinner"></div>
                                    <p>Waiting for detection...</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Upload Tab -->
            <div id="upload-content" class="tab-content">
                <div class="main-grid">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title">
                                <i class="fas fa-upload"></i>
                                Upload Image for Detection
                            </div>
                        </div>
                        
                        <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                            <i class="fas fa-cloud-upload-alt"></i>
                            <h3>Click or Drag to Upload</h3>
                            <p style="color: var(--text-secondary); margin-top: 8px;">
                                Supports: JPG, PNG, JPEG
                            </p>
                        </div>
                        
                        <input type="file" id="fileInput" accept="image/*" onchange="handleFileUpload(event)">
                        
                        <div class="video-wrapper" id="uploadResult" style="display: none;">
                            <canvas id="resultCanvas"></canvas>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title">
                                <i class="fas fa-clipboard-check"></i>
                                Detection Results
                            </div>
                        </div>
                        
                        <div id="uploadStats" class="loading">
                            <p>Upload an image to see results</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>Powered by TensorFlow & AI Technology | Built with ❤️ for Safety</p>
            </div>
        </div>
        
        <script>
            // Audio Context for sound alerts
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            function playSuccessSound() {
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                oscillator.frequency.value = 800;
                oscillator.type = 'sine';
                
                gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
                
                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.3);
            }
            
            function playWarningSound() {
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                oscillator.frequency.value = 400;
                oscillator.type = 'square';
                
                gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
                
                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.5);
            }
            
            // Tab switching
            function switchTab(tab) {
                const tabs = document.querySelectorAll('.tab');
                const contents = document.querySelectorAll('.tab-content');
                
                tabs.forEach(t => t.classList.remove('active'));
                contents.forEach(c => c.classList.remove('active'));
                
                event.target.closest('.tab').classList.add('active');
                document.getElementById(tab + '-content').classList.add('active');
            }
            
            // Real-time detection
            let video = document.getElementById('video');
            let canvas = document.getElementById('canvas');
            let ctx = canvas.getContext('2d');
            let overlayCanvas = document.getElementById('overlayCanvas');
            let overlayCtx = overlayCanvas.getContext('2d');
            let stream = null;
            let isRunning = false;
            let detectionInterval = null;
            let lastSoundTime = 0;
            const soundCooldown = 3000; // 3 seconds
            
            let stats = {
                total: 0,
                withMask: 0,
                withoutMask: 0
            };
            
            async function startCamera() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            facingMode: 'user',
                            width: { ideal: 1280 },
                            height: { ideal: 720 }
                        } 
                    });
                    
                    video.srcObject = stream;
                    isRunning = true;
                    
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('recordingIndicator').classList.add('active');
                    
                    detectionInterval = setInterval(detectFrame, 250);
                    
                } catch (err) {
                    alert('⚠️ Cannot access camera: ' + err.message);
                }
            }
            
            function stopCamera() {
                isRunning = false;
                
                if (detectionInterval) {
                    clearInterval(detectionInterval);
                }
                
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    video.srcObject = null;
                }
                
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('recordingIndicator').classList.remove('active');
                clearOverlay();
            }
            
            async function detectFrame() {
                if (!isRunning) return;
                if (!video.videoWidth || !video.videoHeight) return;
                
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);
                
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                try {
                    const response = await fetch('/detect', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: imageData })
                    });
                    
                    const result = await response.json();
                    updateUI(result);
                } catch (err) {
                    clearOverlay();
                    console.error('Detection error:', err);
                }
            }

            function clearOverlay() {
                overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            }

            function drawRealtimeOverlay(results) {
                overlayCanvas.width = video.videoWidth;
                overlayCanvas.height = video.videoHeight;
                clearOverlay();
                
                results.forEach((r) => {
                    if (!r.bbox) return;
                    
                    const x = r.bbox.x;
                    const y = r.bbox.y;
                    const w = r.bbox.w;
                    const h = r.bbox.h;
                    const color = r.has_mask ? '#00ff00' : '#ff0000';
                    const text = `${r.label}: ${(r.confidence * 100).toFixed(1)}%`;
                    
                    overlayCtx.strokeStyle = color;
                    overlayCtx.lineWidth = 3;
                    overlayCtx.strokeRect(x, y, w, h);
                    
                    overlayCtx.font = 'bold 18px Arial';
                    const textWidth = overlayCtx.measureText(text).width;
                    const textHeight = 24;
                    const boxY = Math.max(0, y - textHeight - 8);
                    
                    overlayCtx.fillStyle = color;
                    overlayCtx.fillRect(x, boxY, textWidth + 14, textHeight + 8);
                    
                    overlayCtx.fillStyle = '#ffffff';
                    overlayCtx.fillText(text, x + 7, boxY + textHeight);
                });
            }
            
            function updateUI(result) {
                drawRealtimeOverlay(result.results || []);

                if (result.faces > 0) {
                    const now = Date.now();
                    
                    result.results.forEach(r => {
                        stats.total++;
                        if (r.has_mask) {
                            stats.withMask++;
                            if (now - lastSoundTime > soundCooldown) {
                                playSuccessSound();
                                lastSoundTime = now;
                            }
                        } else {
                            stats.withoutMask++;
                            if (now - lastSoundTime > soundCooldown) {
                                playWarningSound();
                                lastSoundTime = now;
                            }
                        }
                    });
                    
                    document.getElementById('totalScans').textContent = stats.total;
                    document.getElementById('withMask').textContent = stats.withMask;
                    document.getElementById('withoutMask').textContent = stats.withoutMask;
                    
                    const compliance = stats.total > 0 
                        ? Math.round((stats.withMask / stats.total) * 100) 
                        : 100;
                    document.getElementById('compliance').textContent = compliance + '%';
                    
                    const detectionList = document.getElementById('detectionList');
                    
                    if (detectionList.querySelector('.loading')) {
                        detectionList.innerHTML = '';
                    }
                    
                    result.results.forEach(r => {
                        const item = document.createElement('div');
                        item.className = 'detection-item';
                        
                        const iconClass = r.has_mask ? 'success' : 'danger';
                        const icon = r.has_mask ? 'fa-check-circle' : 'fa-times-circle';
                        
                        item.innerHTML = `
                            <div class="detection-info">
                                <div class="detection-icon ${iconClass}">
                                    <i class="fas ${icon}"></i>
                                </div>
                                <div>
                                    <div class="detection-label">${r.label}</div>
                                    <div class="detection-time">${new Date().toLocaleTimeString()}</div>
                                </div>
                            </div>
                            <div class="confidence-badge">
                                ${(r.confidence * 100).toFixed(1)}%
                            </div>
                        `;
                        
                        detectionList.insertBefore(item, detectionList.firstChild);
                        
                        while (detectionList.children.length > 10) {
                            detectionList.removeChild(detectionList.lastChild);
                        }
                    });
                }
            }
            
            // Upload handling
            const uploadArea = document.getElementById('uploadArea');
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                
                const file = e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) {
                    processUploadedImage(file);
                }
            });
            
            function handleFileUpload(event) {
                const file = event.target.files[0];
                if (file) {
                    processUploadedImage(file);
                }
            }
            
            async function processUploadedImage(file) {
                const reader = new FileReader();
                
                reader.onload = async function(e) {
                    const img = new Image();
                    img.onload = async function() {
                        const resultCanvas = document.getElementById('resultCanvas');
                        const resultCtx = resultCanvas.getContext('2d');
                        
                        resultCanvas.width = img.width;
                        resultCanvas.height = img.height;
                        resultCtx.drawImage(img, 0, 0);
                        
                        const imageData = resultCanvas.toDataURL('image/jpeg');
                        
                        document.getElementById('uploadStats').innerHTML = `
                            <div class="loading">
                                <div class="spinner"></div>
                                <p>Processing image...</p>
                            </div>
                        `;
                        
                        try {
                            const response = await fetch('/detect_upload', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ image: imageData })
                            });
                            
                            const result = await response.json();
                            
                            // Draw results on canvas
                            const annotatedImg = new Image();
                            annotatedImg.onload = function() {
                                resultCtx.drawImage(annotatedImg, 0, 0);
                                document.getElementById('uploadResult').style.display = 'block';
                                resultCanvas.style.display = 'block';
                            };
                            annotatedImg.src = 'data:image/jpeg;base64,' + result.image;
                            
                            // Show stats
                            let statsHTML = '';
                            if (result.faces === 0) {
                                statsHTML = '<p>No faces detected</p>';
                            } else {
                                statsHTML = `
                                    <div class="stats-grid">
                                        <div class="stat-card">
                                            <div class="stat-value">${result.faces}</div>
                                            <div class="stat-label">Faces</div>
                                        </div>
                                        <div class="stat-card success">
                                            <div class="stat-value">${result.with_mask}</div>
                                            <div class="stat-label">With Mask</div>
                                        </div>
                                        <div class="stat-card danger">
                                            <div class="stat-value">${result.without_mask}</div>
                                            <div class="stat-label">Without Mask</div>
                                        </div>
                                        <div class="stat-card">
                                            <div class="stat-value">${result.compliance}%</div>
                                            <div class="stat-label">Compliance</div>
                                        </div>
                                    </div>
                                    <div class="detection-list" style="max-height: 200px;">
                                `;
                                
                                result.results.forEach(r => {
                                    const iconClass = r.has_mask ? 'success' : 'danger';
                                    const icon = r.has_mask ? 'fa-check-circle' : 'fa-times-circle';
                                    
                                    statsHTML += `
                                        <div class="detection-item">
                                            <div class="detection-info">
                                                <div class="detection-icon ${iconClass}">
                                                    <i class="fas ${icon}"></i>
                                                </div>
                                                <div>
                                                    <div class="detection-label">${r.label}</div>
                                                    <div class="detection-time">Quality: ${r.quality_score.toFixed(0)}</div>
                                                </div>
                                            </div>
                                            <div class="confidence-badge">
                                                ${(r.confidence * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                    `;
                                });
                                
                                statsHTML += '</div>';
                                
                                // Play sound
                                if (result.without_mask > 0) {
                                    playWarningSound();
                                } else if (result.with_mask > 0) {
                                    playSuccessSound();
                                }
                            }
                            
                            document.getElementById('uploadStats').innerHTML = statsHTML;
                            
                        } catch (err) {
                            console.error('Upload detection error:', err);
                            document.getElementById('uploadStats').innerHTML = 
                                '<p style="color: var(--danger);">Error processing image</p>';
                        }
                    };
                    img.src = e.target.result;
                };
                
                reader.readAsDataURL(file);
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/detect', methods=['POST'])
def detect():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image_array = np.array(image)
        
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        
        _, results = detect_mask_advanced(image_bgr, threshold=None, min_confidence=0.75)
        
        stats['total_scans'] += len(results)
        for r in results:
            if r['has_mask']:
                stats['with_mask'] += 1
            else:
                stats['without_mask'] += 1
        stats['last_detection'] = datetime.now().isoformat()
        
        return jsonify({
            'faces': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect_upload', methods=['POST'])
def detect_upload():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image_array = np.array(image)
        
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        
        annotated_image, results = detect_mask_advanced(image_bgr, threshold=None, min_confidence=0.60)
        
        # Encode result image
        _, buffer = cv2.imencode('.jpg', annotated_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        with_mask = sum(1 for r in results if r['has_mask'])
        without_mask = len(results) - with_mask
        compliance = round((with_mask / len(results) * 100) if len(results) > 0 else 100)
        
        return jsonify({
            'faces': len(results),
            'results': results,
            'with_mask': with_mask,
            'without_mask': without_mask,
            'compliance': compliance,
            'image': image_base64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': MODEL_LOADED}), 200

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
