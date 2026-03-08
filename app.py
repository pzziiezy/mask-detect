import os
import sys
import cv2
import numpy as np
import base64
import json
from flask import Flask, render_template_string, request, jsonify
from tensorflow import keras
from io import BytesIO
from PIL import Image
from datetime import datetime
from collections import OrderedDict

#  MediaPipe ของ Google
import mediapipe as mp

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILENAME = 'mask_detector_production.h5' 
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

app = Flask(__name__)

# --- GLOBAL VARIABLES ---
model = None
MODEL_LOADED = False
SYSTEM_ERROR_LOG = ""

#  MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=1, # 🛠️ เปลี่ยนเป็น 1: เหมาะกับวิดีโอจำลองคนเดินไกลๆ
    min_detection_confidence=0.60 # 🛠️ เพิ่มเป็น 50% ป้องกันการตีกรอบมั่ว
)

stats = {'total_detections': 0, 'total_mask': 0, 'total_no_mask': 0}
object_history = {}

# --- TRACKER CLASS (อัปเกรดเป็น 4D: จำตำแหน่ง + ขนาดใบหน้า) ---
class SimpleTracker:
    def __init__(self, maxDisappeared=2, maxDistance=100):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, features):
        self.objects[self.nextObjectID] = features
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        return self.nextObjectID - 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        deregistered_ids = []
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    deregistered_ids.append(objectID)
                    self.deregister(objectID)
            return [], deregistered_ids

        # จุดอัปเกรด 4D: เก็บค่า X, Y และเพิ่ม ความกว้าง (W), ความสูง (H)
        inputFeatures = np.zeros((len(rects), 4), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            w = endX - startX
            h = endY - startY
            inputFeatures[i] = (cX, cY, w, h)

        if len(self.objects) == 0:
            assignments = []
            for i in range(0, len(inputFeatures)):
                obj_id = self.register(inputFeatures[i])
                assignments.append((obj_id, i))
            return assignments, deregistered_ids

        objectIDs = list(self.objects.keys())
        objectFeatures = list(self.objects.values())

        # คำนวณความแตกต่างทั้ง 4 มิติ
        D = np.linalg.norm(np.array(objectFeatures)[:, np.newaxis] - inputFeatures, axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()
        assignments = []

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols: continue
            if D[row, col] > self.maxDistance: continue

            objectID = objectIDs[row]
            self.objects[objectID] = inputFeatures[col]
            self.disappeared[objectID] = 0
            assignments.append((objectID, col))
            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                deregistered_ids.append(objectID)
                self.deregister(objectID)

        unusedCols = set(range(0, D.shape[1])).difference(usedCols)
        for col in unusedCols:
            obj_id = self.register(inputFeatures[col])
            assignments.append((obj_id, col))
            
        return assignments, deregistered_ids

# ให้ลืมทันทีใน 2 เฟรม และถ้าระยะหน้าวาร์ปเกิน 75 พิกเซล
tracker = SimpleTracker(maxDisappeared=2, maxDistance=75)

# --- INITIALIZATION ---
def initialize_system():
    global model, MODEL_LOADED, SYSTEM_ERROR_LOG
    print("="*50)
    try:
        import keras
        orig_dense = keras.layers.Dense.__init__
        def safe_dense(self, *args, **kwargs):
            kwargs.pop('quantization_config', None) 
            orig_dense(self, *args, **kwargs)
        keras.layers.Dense.__init__ = safe_dense

        orig_conv2d = keras.layers.Conv2D.__init__
        def safe_conv2d(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            orig_conv2d(self, *args, **kwargs)
        keras.layers.Conv2D.__init__ = safe_conv2d
    except Exception:
        pass

    if os.path.exists(MODEL_PATH):
        try:
            model = keras.models.load_model(MODEL_PATH, compile=False)
            MODEL_LOADED = True
        except Exception as e:
            SYSTEM_ERROR_LOG += f"[Model Load Error: {str(e)}] "
    
    print("="*50)

initialize_system()

# --- WEB UI (HTML + JS) ---
@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Mask Surveillance Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-base: #0a0e17;
            --panel-bg: rgba(16, 22, 36, 0.85);
            --border-color: #1e293b;
            --text-main: #e2e8f0;
            --text-muted: #94a3b8;
            --neon-blue: #3b82f6;
            --neon-green: #10b981;
            --neon-red: #ef4444;
        }

        body { 
            background-color: var(--bg-base); 
            background-image: radial-gradient(circle at 50% 0%, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
            color: var(--text-main); 
            font-family: 'Inter', sans-serif; 
            min-height: 100vh;
        }

        h1, h2, h3, h4, .navbar-brand { font-family: 'Rajdhani', sans-serif; font-weight: 700; letter-spacing: 1px; }
        
        .navbar { 
            background: rgba(10, 14, 23, 0.95) !important; 
            border-bottom: 1px solid var(--border-color);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
        }
        .live-indicator { display: inline-block; width: 10px; height: 10px; background-color: var(--neon-red); border-radius: 50%; box-shadow: 0 0 10px var(--neon-red); animation: pulse 1.5s infinite; margin-right: 8px; }

        .glass-panel {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }

        .video-wrapper { 
            position: relative; 
            border-radius: 12px; 
            overflow: hidden; 
            background: #000; 
            border: 2px solid var(--border-color);
            transition: all 0.3s ease; 
            aspect-ratio: 16 / 9; /*  ล็อกสัดส่วนจอให้กางรอไว้เลย */
        }
        .video-wrapper.active { border-color: var(--neon-blue); box-shadow: 0 0 25px rgba(59, 130, 246, 0.2); }
        .video-wrapper.alert-flash { border-color: var(--neon-red); box-shadow: 0 0 35px rgba(239, 68, 68, 0.6); }
        video { width: 100%; height: 100%; object-fit: cover; display: block; transform: scaleX(-1); }
        
        .fps-badge { position: absolute; top: 12px; left: 12px; z-index: 10; background: rgba(0,0,0,0.7); border: 1px solid var(--neon-green); color: var(--neon-green); padding: 4px 10px; border-radius: 6px; font-family: 'Rajdhani', monospace; font-size: 16px; font-weight: bold; box-shadow: 0 0 10px rgba(16, 185, 129, 0.2); }

        .kpi-card { padding: 20px; text-align: center; height: 100%; transition: transform 0.2s; }
        .kpi-card:hover { transform: translateY(-2px); }
        .kpi-title { font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1.5px; margin-bottom: 5px; }
        .kpi-value { font-size: 2.5rem; line-height: 1; }
        .text-glow-blue { color: var(--neon-blue); text-shadow: 0 0 15px rgba(59,130,246,0.4); }
        .text-glow-green { color: var(--neon-green); text-shadow: 0 0 15px rgba(16,185,129,0.4); }
        .text-glow-red { color: var(--neon-red); text-shadow: 0 0 15px rgba(239,68,68,0.4); }

        .log-box { height: 380px; overflow-y: auto; }
        .table { color: var(--text-main); margin-bottom: 0; }
        .table-light { background-color: rgba(30, 41, 59, 0.9); color: var(--text-main); }
        .table-light th { border-bottom: 1px solid var(--border-color); color: var(--text-muted); font-weight: 500; letter-spacing: 1px; }
        .table-hover tbody tr:hover { background-color: rgba(59, 130, 246, 0.1); color: #fff; }
        td { border-bottom: 1px solid var(--border-color) !important; padding: 12px 8px !important; }
        
        .badge-mask { background-color: rgba(16, 185, 129, 0.15); border: 1px solid var(--neon-green); color: var(--neon-green); box-shadow: 0 0 8px rgba(16, 185, 129, 0.2); }
        .badge-nomask { background-color: rgba(239, 68, 68, 0.15); border: 1px solid var(--neon-red); color: var(--neon-red); box-shadow: 0 0 8px rgba(239, 68, 68, 0.2); }
        .snapshot-img { width: 50px; height: 50px; object-fit: cover; border-radius: 8px; border: 2px solid #334155; }
        
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-base); }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--neon-blue); }

        @keyframes pulse { 0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); } 70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); } 100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } }
        .new-entry { animation: highlightRow 1.5s ease-out; }
        @keyframes highlightRow { 0% { background-color: rgba(59, 130, 246, 0.2); } 100% { background-color: transparent; } }

        .form-select { background-color: #1e293b; color: #fff; border: 1px solid #334155; }
        .form-select:focus { background-color: #1e293b; color: #fff; border-color: var(--neon-blue); box-shadow: none; }
        .btn-glow { position: relative; overflow: hidden; }
        .btn-glow::after { content: ''; position: absolute; top: 50%; left: 50%; width: 120%; height: 120%; background: radial-gradient(circle, rgba(255,255,255,0.2) 10%, transparent 10.01%); transform: translate(-50%, -50%) scale(10); opacity: 0; transition: transform 0.5s, opacity 0.5s; }
        .btn-glow:active::after { transform: translate(-50%, -50%) scale(0); opacity: 1; transition: 0s; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark mb-4 sticky-top">
        <div class="container-fluid px-4">
            <span class="navbar-brand mb-0 h2 d-flex align-items-center">
                <i class="fas fa-shield-virus text-primary me-3" style="font-size: 1.5rem;"></i> 
                AI MASK SURVEILLANCE
            </span>
            <div class="d-flex align-items-center">
                <div class="d-flex align-items-center me-4">
                    <span class="live-indicator" id="live-dot" style="display: none;"></span>
                    <span class="text-muted small fw-bold" id="status-text">SYSTEM STANDBY</span>
                </div>
                <select id="camera-select" class="form-select form-select-sm me-3" style="width: 200px;">
                    <option value="">Initializing Video Input...</option>
                </select>
                <button class="btn btn-outline-info btn-sm" onclick="exportCSV()"><i class="fas fa-file-csv"></i> Export Data</button>
            </div>
        </div>
    </nav>

    <div class="container-fluid px-4 pb-4">
        <div class="row g-4">
            <div class="col-lg-7">
                <div class="glass-panel p-3 mb-3">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h5 class="m-0"><i class="fas fa-video text-muted me-2"></i> Live Feed</h5>
                    </div>
                    <div class="video-wrapper mb-3" id="video-container">
                        <div class="fps-badge" id="fps-display">FPS: 0</div>
                        <video id="video" autoplay playsinline></video>
                        <canvas id="canvas"></canvas>
                    </div>
                    <div class="d-flex justify-content-center gap-3">
                        <button class="btn btn-primary px-5 py-2 fw-bold btn-glow" onclick="startCamera()"><i class="fas fa-power-off me-2"></i> ACTIVATE SYSTEM</button>
                        <button class="btn btn-outline-danger px-4 py-2 fw-bold" onclick="stopCamera()"><i class="fas fa-stop me-2"></i> HALT</button>
                    </div>
                </div>

                <div class="glass-panel p-3">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h6 class="m-0 text-muted"><i class="fas fa-wave-square me-2 text-primary"></i> Real-Time Detection Trend (Live)</h6>
                    </div>
                    <div style="position: relative; height: 180px; width: 100%;">
                        <canvas id="trendChart"></canvas>
                    </div>
                </div>
            </div>

            <div class="col-lg-5">
                <div class="row g-3 mb-3">
                    <div class="col-sm-4">
                        <div class="glass-panel kpi-card" style="border-bottom: 3px solid var(--neon-blue);">
                            <div class="kpi-title">Total Scans</div>
                            <h2 class="fw-bold text-glow-blue mb-0" id="total-count">0</h2>
                        </div>
                    </div>
                    <div class="col-sm-4">
                        <div class="glass-panel kpi-card" style="border-bottom: 3px solid var(--neon-green);">
                            <div class="kpi-title">Compliance</div>
                            <h2 class="fw-bold text-glow-green mb-0" id="mask-count">0</h2>
                        </div>
                    </div>
                    <div class="col-sm-4">
                        <div class="glass-panel kpi-card" style="border-bottom: 3px solid var(--neon-red);">
                            <div class="kpi-title">Violations</div>
                            <h2 class="fw-bold text-glow-red mb-0" id="nomask-count">0</h2>
                        </div>
                    </div>
                </div>

                <div class="glass-panel p-3 mb-3" style="height: 220px;">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h6 class="m-0 text-muted"><i class="fas fa-chart-pie me-2"></i> Distribution Ratio</h6>
                    </div>
                    <div style="position: relative; height: 85%; width: 100%;">
                        <canvas id="statsChart"></canvas>
                    </div>
                </div>

                <div class="glass-panel log-box">
                    <div class="p-3 border-bottom d-flex justify-content-between align-items-center sticky-top" style="background: rgba(16, 22, 36, 0.95); backdrop-filter: blur(10px); z-index: 5; border-color: var(--border-color) !important;">
                        <h6 class="fw-bold m-0"><i class="fas fa-list-ul text-primary me-2"></i> Detection Evidence Log</h6>
                        <button class="btn btn-sm btn-outline-secondary" onclick="clearLogs()"><i class="fas fa-eraser"></i> Purge</button>
                    </div>
                    <div class="table-responsive">
                        <table class="table table-hover align-middle mb-0" id="report-table">
                            <thead class="table-light">
                                <tr>
                                    <th class="ps-4">Target</th>
                                    <th>Timestamp / Ref</th>
                                    <th class="text-end pe-4">Classification</th>
                                </tr>
                            </thead>
                            <tbody id="log-body"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const videoContainer = document.getElementById('video-container');
        
        let stream, isRunning = false;
        let isProcessing = false; 
        let lastFrameTime = Date.now();
        let trendInterval; // เก็บตัวจับเวลาเพื่อวาดกราฟ Trend

        const AudioContext = window.AudioContext || window.webkitAudioContext;
        const audioCtx = new AudioContext();

        // Dark theme Chart.js defaults
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = 'Inter';

        // Doughnut Chart
        const chartCtx = document.getElementById('statsChart').getContext('2d');
        let statsChart = new Chart(chartCtx, {
            type: 'doughnut',
            data: {
                labels: ['Compliant (Mask)', 'Violation (No Mask)'],
                datasets: [{ 
                    data: [0, 0], 
                    backgroundColor: ['#10b981', '#ef4444'], 
                    borderColor: '#1e293b',
                    borderWidth: 2,
                    hoverOffset: 4
                }]
            },
            options: { 
                responsive: true, 
                maintainAspectRatio: false, 
                cutout: '75%', 
                plugins: { 
                    legend: { position: 'right', labels: { usePointStyle: true, padding: 20 } } 
                } 
            }
        });

        // 🚀 NEW: Trend Line Chart
        const trendCtx = document.getElementById('trendChart').getContext('2d');
        let trendChart = new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: [], 
                datasets: [
                    { 
                        label: 'Mask (Cumulative)', 
                        data: [], 
                        borderColor: '#10b981', 
                        backgroundColor: 'rgba(16, 185, 129, 0.15)', 
                        fill: true, 
                        tension: 0.4, 
                        borderWidth: 2, 
                        pointRadius: 0 
                    },
                    { 
                        label: 'No Mask (Cumulative)', 
                        data: [], 
                        borderColor: '#ef4444', 
                        backgroundColor: 'rgba(239, 68, 68, 0.15)', 
                        fill: true, 
                        tension: 0.4, 
                        borderWidth: 2, 
                        pointRadius: 0 
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 0 }, 
                scales: {
                    x: { display: true, grid: { color: 'rgba(30, 41, 59, 0.5)' }, ticks: { maxTicksLimit: 10 } },
                    y: { display: true, beginAtZero: true, grid: { color: 'rgba(30, 41, 59, 0.5)' } }
                },
                plugins: { 
                    legend: { display: false },
                    tooltip: { mode: 'index', intersect: false }
                }
            }
        });

        // 🚀 ฟังก์ชันอัปเดตกราฟเส้นทุกๆ 1 วินาที
        function startTrendRecording() {
            trendInterval = setInterval(() => {
                if(!isRunning) return;
                
                const now = new Date();
                const timeStr = now.getHours().toString().padStart(2,'0') + ':' + 
                                now.getMinutes().toString().padStart(2,'0') + ':' + 
                                now.getSeconds().toString().padStart(2,'0');
                
                const currentMask = parseInt(document.getElementById('mask-count').innerText);
                const currentNoMask = parseInt(document.getElementById('nomask-count').innerText);

                trendChart.data.labels.push(timeStr);
                trendChart.data.datasets[0].data.push(currentMask);
                trendChart.data.datasets[1].data.push(currentNoMask);
                
                // เลื่อนกราฟไปข้างหน้า (เก็บ 30 วินาทีล่าสุด)
                if(trendChart.data.labels.length > 30) { 
                    trendChart.data.labels.shift();
                    trendChart.data.datasets[0].data.shift();
                    trendChart.data.datasets[1].data.shift();
                }
                trendChart.update();
            }, 1000);
        }

        window.onload = async function() {
            clearLogs();
            await populateCameras();
        };

        async function populateCameras() {
            try {
                await navigator.mediaDevices.getUserMedia({ audio: false, video: true }); 
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoInputs = devices.filter(device => device.kind === 'videoinput');
                const select = document.getElementById('camera-select');
                select.innerHTML = '';
                videoInputs.forEach((device, index) => {
                    const option = document.createElement('option');
                    option.value = device.deviceId;
                    option.text = device.label || `Camera ${index + 1}`;
                    select.appendChild(option);
                });
                select.onchange = () => { if(isRunning){ stopCamera(); startCamera(); } };
            } catch (e) { console.warn("Camera enumeration failed:", e); }
        }

        function playAlertSound() {
            if(audioCtx.state === 'suspended') audioCtx.resume();
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.type = 'square';
            osc.frequency.value = 400; 
            gain.gain.setValueAtTime(0.05, audioCtx.currentTime);
            osc.start();
            osc.stop(audioCtx.currentTime + 0.2); 
        }

        async function startCamera() {
            if (isRunning) return;
            const deviceId = document.getElementById('camera-select').value;
            const constraints = { video: { width: {ideal: 640}, height: {ideal: 480} } };
            if (deviceId) constraints.video.deviceId = { exact: deviceId };
            else constraints.video.facingMode = 'user';

            try {
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    video.play(); //  บังคับให้เบราว์เซอร์เล่นภาพ!
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    isRunning = true;
                    videoContainer.classList.add('active');
                    document.getElementById('live-dot').style.display = 'inline-block';
                    document.getElementById('status-text').innerText = 'SYSTEM ACTIVE';
                    document.getElementById('status-text').className = 'small fw-bold text-success';
                    
                    startTrendRecording(); // เริ่มบันทึกกราฟเส้น
                    sendFrame(); 
                };
            } catch (err) { alert(err.message); }
        }

        function stopCamera() {
            if(stream) stream.getTracks().forEach(t=>t.stop());
            ctx.clearRect(0,0,canvas.width,canvas.height);
            isRunning = false;
            isProcessing = false; 
            clearInterval(trendInterval); // หยุดบันทึกกราฟ
            videoContainer.classList.remove('active');
            document.getElementById('live-dot').style.display = 'none';
            document.getElementById('status-text').innerText = 'SYSTEM STANDBY';
            document.getElementById('status-text').className = 'text-muted small fw-bold';
            document.getElementById('fps-display').innerText = "FPS: 0";
        }

        function exportCSV() {
            const tbody = document.getElementById('log-body');
            const rows = tbody.querySelectorAll('tr');
            let csvContent = "Scan ID,Time,Result,Confidence (%),Note\\n";
            rows.forEach(row => {
                try {
                    let scanIdElem = row.querySelector('.fw-bold.text-light');
                    let timeElem = row.querySelector('.text-muted');
                    let resultElem = row.querySelector('.badge');
                    let confElem = row.querySelector('.conf-val'); 
                    let id = scanIdElem ? scanIdElem.innerText.replace('REF-', '').trim() : '';
                    let time = timeElem ? timeElem.innerText.trim() : '';
                    let result = resultElem ? resultElem.innerText.trim() : '';
                    let conf = confElem ? confElem.innerText.replace('%', '').trim() : '';
                    let isChange = row.innerHTML.includes('Status Updated') ? 'Changed' : 'New';
                    csvContent += `${id},${time},${result},${conf},${isChange}\\n`;
                } catch(e) { console.error("Error parsing row for CSV:", e); }
            });
            const blob = new Blob(["\\uFEFF" + csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.setAttribute("href", url);
            link.setAttribute("download", `Security_Log_${new Date().toISOString().split('T')[0]}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        function sendFrame() {
            if (!isRunning || isProcessing) return; 
            isProcessing = true; 

            const temp = document.createElement('canvas');
            temp.width = video.videoWidth; temp.height = video.videoHeight;
            temp.getContext('2d').drawImage(video,0,0);
            
            fetch('/detect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image: temp.toDataURL('image/jpeg', 0.6)})
            })
            .then(r=>r.json()).then(data => {
                if(data.error) return;

                const now = Date.now();
                const delta = now - lastFrameTime;
                if(delta > 0) { document.getElementById('fps-display').innerText = `FPS: ${Math.round(1000/delta)}`; }
                lastFrameTime = now;

                document.getElementById('total-count').innerText = data.stats.total_detections;
                document.getElementById('mask-count').innerText = data.stats.total_mask;
                document.getElementById('nomask-count').innerText = data.stats.total_no_mask;
                
                statsChart.data.datasets[0].data = [data.stats.total_mask, data.stats.total_no_mask];
                statsChart.update();

                ctx.clearRect(0,0,canvas.width,canvas.height);
                
                if(!data.results) return;

                data.results.forEach(face => {
                    const [x,y,w,h] = face.bbox;
                    const color = face.has_mask ? '#10b981' : '#ef4444'; // Neon Green / Red
                    const mirroredX = canvas.width - x - w; 
                    
                    // Draw futuristic bounding box
                    ctx.strokeStyle = color; 
                    ctx.lineWidth = 2;
                    ctx.strokeRect(mirroredX, y, w, h);
                    
                    // Corner accents
                    const clen = 15;
                    ctx.lineWidth = 4;
                    ctx.beginPath();
                    ctx.moveTo(mirroredX, y+clen); ctx.lineTo(mirroredX, y); ctx.lineTo(mirroredX+clen, y);
                    ctx.moveTo(mirroredX+w-clen, y); ctx.lineTo(mirroredX+w, y); ctx.lineTo(mirroredX+w, y+clen);
                    ctx.moveTo(mirroredX+w, y+h-clen); ctx.lineTo(mirroredX+w, y+h); ctx.lineTo(mirroredX+w-clen, y+h);
                    ctx.moveTo(mirroredX+clen, y+h); ctx.lineTo(mirroredX, y+h); ctx.lineTo(mirroredX, y+h-clen);
                    ctx.stroke();
                    
                    // Label
                    ctx.fillStyle = "rgba(0,0,0,0.7)";
                    const text = `${face.label} ${face.confidence}%`;
                    ctx.font = "bold 14px 'Rajdhani', sans-serif";
                    const textWidth = ctx.measureText(text).width;
                    ctx.fillRect(mirroredX, y-28, textWidth + 16, 26);
                    
                    ctx.fillStyle = color;
                    ctx.fillText(text, mirroredX + 8, y - 9);
                });

                if (data.new_logs?.length > 0) {
                    const tbody = document.getElementById('log-body');
                    let triggerAlert = false;

                    data.new_logs.forEach(log => {
                        if (!log.has_mask) triggerAlert = true;

                        const tr = document.createElement('tr');
                        tr.className = 'new-entry';
                        
                        const badgeClass = log.has_mask ? 'badge-mask' : 'badge-nomask';
                        const badgeText = log.has_mask ? 'COMPLIANT' : 'VIOLATION';
                        const textClass = log.has_mask ? 'text-success' : 'text-danger';
                        
                        const badge = `<span class="badge ${badgeClass} d-block mb-1 py-2" style="font-family: 'Rajdhani'; letter-spacing: 1px;">${badgeText}</span>
                                       <small class="${textClass} fw-bold"><i class="fas fa-crosshairs"></i> <span class="conf-val">${log.confidence}%</span></small>`;
                        
                        const changedText = log.is_change ? '<span class="text-warning" style="font-size: 0.7rem;"><br><i class="fas fa-sync-alt fa-spin"></i> Status Updated</span>' : '';
                        const imgSrc = log.snapshot ? `data:image/jpeg;base64,${log.snapshot}` : 'https://via.placeholder.com/50?text=?';

                        tr.innerHTML = `
                            <td class="ps-4"><img src="${imgSrc}" class="snapshot-img"></td>
                            <td>
                                <div class="fw-bold text-light" style="font-family: 'Rajdhani'; font-size: 1.1rem;">REF-${log.scan_num.toString().padStart(4, '0')}</div>
                                <small class="text-muted"><i class="far fa-clock"></i> ${log.time}</small>${changedText}
                            </td>
                            <td class="text-end pe-4 align-middle">${badge}</td>
                        `;
                        tbody.insertBefore(tr, tbody.firstChild);
                    });

                    if (triggerAlert) {
                        playAlertSound();
                        videoContainer.classList.add('alert-flash');
                        setTimeout(() => videoContainer.classList.remove('alert-flash'), 600);
                    }
                }
            })
            .catch(e => console.error("Fetch Error:", e))
            .finally(() => {
                isProcessing = false;
                if (isRunning) {
                    setTimeout(sendFrame, 100); 
                }
            });
        }

        function clearLogs() {
            fetch('/clear_logs').then(() => {
                document.getElementById('log-body').innerHTML = '';
                document.getElementById('total-count').innerText = '0';
                document.getElementById('mask-count').innerText = '0';
                document.getElementById('nomask-count').innerText = '0';
                
                statsChart.data.datasets[0].data = [0, 0];
                statsChart.update();

                // 🚀 ล้างกราฟเส้นด้วยเวลาเคลียร์ Log
                trendChart.data.labels = [];
                trendChart.data.datasets[0].data = [];
                trendChart.data.datasets[1].data = [];
                trendChart.update();
            });
        }
    </script>
</body>
</html>
    """)

@app.route('/detect', methods=['POST'])
def detect():
    global stats, object_history
    
    if not MODEL_LOADED: 
        return jsonify({'error': f'System Initialization Failed. REASON: {SYSTEM_ERROR_LOG}'}), 500

    try:
        data = request.json
        image = Image.open(BytesIO(base64.b64decode(data['image'].split(',')[1])))
        
        image_rgb = np.array(image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) 
        
        mp_results = face_detector.process(image_rgb)
        
        results = []
        rects = [] 
        
        if mp_results.detections:
            img_h, img_w, _ = image_rgb.shape
            
            for detection in mp_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * img_w)
                y = int(bbox.ymin * img_h)
                w = int(bbox.width * img_w)
                h = int(bbox.height * img_h)
                
                if w < 30 or h < 30:
                    continue
                
                x = max(0, x)
                y = max(0, y)
                w = min(img_w - x, w)
                h = min(img_h - y, h)
                
                rects.append((x, y, x+w, y+h))
                
                pad_x = int(w * 0.10)
                pad_y = int(h * 0.30)
                startX = max(0, x - pad_x)
                startY = max(0, y - pad_y)
                endX = min(img_w, x + w + pad_x)
                endY = min(img_h, y + h + pad_y)
                
                face_img = image_bgr[startY:endY, startX:endX]
                if face_img.size == 0: continue
                
                _, buffer = cv2.imencode('.jpg', face_img)
                face_b64 = base64.b64encode(buffer).decode('utf-8')

                resized = cv2.resize(face_img, (224, 224))
                normalized = resized.astype('float32') / 255.0
                batch = np.expand_dims(normalized, axis=0)
                
                preds = model.predict(batch, verbose=0)[0]
                
                if isinstance(preds, np.ndarray) and len(preds) > 1:
                    has_mask = preds[0] > preds[1]  
                    confidence = float(preds[0]) if has_mask else float(preds[1])
                else:
                    has_mask = preds < 0.5
                    confidence = float(1.0 - preds) if has_mask else float(preds)

                conf_percent = int(confidence * 100)
                label = "Mask" if has_mask else "No Mask"
                
                results.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'has_mask': bool(has_mask),
                    'label': label,
                    'confidence': conf_percent,
                    'snapshot': face_b64
                })

        active_assignments, deregistered_ids = tracker.update(rects)
        new_logs = []
        current_time = datetime.now().strftime("%H:%M:%S")

        # 1. ปรับเกณฑ์ความนิ่งของ Log (จาก 2 เป็น 5 เฟรม)
        STABILITY_THRESHOLD = 5 

        for (obj_id, idx) in active_assignments:
            if idx < len(results):
                r = results[idx]
                
                if obj_id not in object_history:
                    # 2. เพิ่ม 'history' ลงไปเพื่อใช้เก็บประวัติย้อนหลัง
                    object_history[obj_id] = {'state': None, 'mask_streak': 0, 'nomask_streak': 0, 'history': []}
                
                # --- 3. ระบบลดการกะพริบของกรอบ (Smooth Bounding Box) ---
                object_history[obj_id]['history'].append(r['has_mask'])
                if len(object_history[obj_id]['history']) > 5:
                    object_history[obj_id]['history'].pop(0) # เก็บประวัติแค่ 5 เฟรมล่าสุด
                
                # หาค่าเฉลี่ย (เสียงข้างมาก) ใน 5 เฟรมล่าสุด
                mask_votes = sum(object_history[obj_id]['history'])
                stable_has_mask = mask_votes >= 3 # ถ้าโหวตได้ >= 3 ใน 5 ถือว่าใส่แมส
                
                r['has_mask'] = stable_has_mask
                r['label'] = "Mask" if stable_has_mask else "No Mask"
                # -----------------------------------------------------------

                if stable_has_mask:
                    object_history[obj_id]['mask_streak'] += 1
                    object_history[obj_id]['nomask_streak'] = 0
                else:
                    object_history[obj_id]['nomask_streak'] += 1
                    object_history[obj_id]['mask_streak'] = 0

                if object_history[obj_id]['mask_streak'] >= STABILITY_THRESHOLD and object_history[obj_id]['state'] != 'Mask':
                    is_change = object_history[obj_id]['state'] is not None
                    object_history[obj_id]['state'] = 'Mask'
                    
                    stats['total_detections'] += 1
                    stats['total_mask'] += 1
                    
                    new_logs.append({
                        'id': obj_id,
                        'scan_num': stats['total_detections'],
                        'time': current_time,
                        'has_mask': True,
                        'is_change': is_change,
                        'confidence': r['confidence'],
                        'snapshot': r['snapshot'] 
                    })

                elif object_history[obj_id]['nomask_streak'] >= STABILITY_THRESHOLD and object_history[obj_id]['state'] != 'No Mask':
                    is_change = object_history[obj_id]['state'] is not None
                    object_history[obj_id]['state'] = 'No Mask'
                    
                    stats['total_detections'] += 1
                    stats['total_no_mask'] += 1
                    
                    new_logs.append({
                        'id': obj_id,
                        'scan_num': stats['total_detections'],
                        'time': current_time,
                        'has_mask': False,
                        'is_change': is_change,
                        'confidence': r['confidence'],
                        'snapshot': r['snapshot'] 
                    })

        for obj_id in deregistered_ids:
            if obj_id in object_history:
                del object_history[obj_id]

        for res in results:
            res.pop('snapshot', None)

        return jsonify({'results': results, 'stats': stats, 'new_logs': new_logs})

    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/clear_logs')
def clear_logs():
    global stats, tracker, object_history
    stats = {'total_detections': 0, 'total_mask': 0, 'total_no_mask': 0}
    object_history = {}
    tracker = SimpleTracker(maxDisappeared=2, maxDistance=75)
    return "OK", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"🚀 MobileNetV2 + MediaPipe System running at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)