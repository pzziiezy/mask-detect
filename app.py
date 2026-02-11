import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
import io

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="😷 Mask Detection",
    page_icon="😷",
    layout="wide"
)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# โหลดโมเดล
@st.cache_resource
def load_model():
    try:
        return keras.models.load_model('models/mask_detector.h5')
    except Exception as e:
        st.error(f"Cannot load model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    st.error(f"Cannot load face detector: {e}")
    st.stop()

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Detection", "📊 Reports"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.3, 0.05)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Model Accuracy:** 99.80%\n\n"
    "**Dataset:** 12K Images\n\n"
    "📱 **Mobile Compatible**"
)

# ฟังก์ชันตรวจจับ
def detect_mask(image, threshold=0.3):
    IMG_SIZE = 128
    labels = ['❌ No Mask', '✅ With Mask']
    colors = [(255, 0, 0), (0, 255, 0)]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    results = []
    
    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        
        prediction = model.predict(face_img, verbose=0)[0][0]
        label_idx = 1 if prediction > threshold else 0
        label = labels[label_idx]
        confidence = prediction if label_idx == 1 else (1 - prediction)
        color = colors[label_idx]
        
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
        text = f"{label}: {confidence*100:.1f}%"
        
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (x, y-35), (x + text_width + 10, y), color, -1)
        cv2.putText(image, text, (x+5, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        results.append({
            'label': label,
            'confidence': confidence,
            'has_mask': label_idx == 1
        })
    
    return image, results, len(faces)

# Real-time Camera HTML Component
def realtime_camera_component():
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .video-container {{
                position: relative;
                max-width: 100%;
                margin: 0 auto;
            }}
            #video {{
                width: 100%;
                max-width: 640px;
                border: 3px solid #4CAF50;
                border-radius: 10px;
                display: block;
                margin: 0 auto;
            }}
            .controls {{
                text-align: center;
                margin: 20px 0;
            }}
            .btn {{
                background-color: #4CAF50;
                color: white;
                padding: 12px 24px;
                font-size: 16px;
                margin: 5px;
                cursor: pointer;
                border: none;
                border-radius: 8px;
                transition: 0.3s;
            }}
            .btn:hover {{
                background-color: #45a049;
            }}
            .btn-stop {{
                background-color: #f44336;
            }}
            .btn-stop:hover {{
                background-color: #da190b;
            }}
            .stats {{
                text-align: center;
                margin: 15px 0;
                font-size: 18px;
                font-weight: bold;
            }}
            .status {{
                padding: 10px;
                border-radius: 5px;
                margin: 10px auto;
                max-width: 640px;
            }}
            .status-good {{
                background-color: #d4edda;
                color: #155724;
            }}
            .status-warning {{
                background-color: #fff3cd;
                color: #856404;
            }}
            .status-danger {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            canvas {{
                display: none;
            }}
        </style>
    </head>
    <body>
        <div class="video-container">
            <video id="video" autoplay playsinline></video>
            <canvas id="canvas"></canvas>
        </div>
        
        <div class="controls">
            <button class="btn" id="startBtn" onclick="startCamera()">📹 Start Detection</button>
            <button class="btn btn-stop" id="stopBtn" onclick="stopCamera()" style="display:none;">⏹️ Stop</button>
        </div>
        
        <div id="status" class="status" style="display:none;"></div>
        
        <script>
            let stream = null;
            let isRunning = false;
            let intervalId = null;
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const statusDiv = document.getElementById('status');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            
            async function startCamera() {{
                try {{
                    stream = await navigator.mediaDevices.getUserMedia({{ 
                        video: {{ 
                            facingMode: 'user',
                            width: {{ ideal: 640 }},
                            height: {{ ideal: 480 }}
                        }} 
                    }});
                    
                    video.srcObject = stream;
                    await video.play();
                    
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    
                    isRunning = true;
                    startBtn.style.display = 'none';
                    stopBtn.style.display = 'inline-block';
                    statusDiv.style.display = 'block';
                    
                    // เริ่ม capture frames ทุก 500ms
                    intervalId = setInterval(captureAndSend, 500);
                    
                }} catch (error) {{
                    alert('❌ Cannot access camera: ' + error.message);
                }}
            }}
            
            function captureAndSend() {{
                if (!isRunning) return;
                
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                // ส่งไปยัง Streamlit
                window.parent.postMessage({{
                    isStreamlitMessage: true,
                    type: 'streamlit:setComponentValue',
                    key: 'realtime_camera',
                    value: imageData
                }}, '*');
            }}
            
            function stopCamera() {{
                isRunning = false;
                
                if (intervalId) {{
                    clearInterval(intervalId);
                    intervalId = null;
                }}
                
                if (stream) {{
                    stream.getTracks().forEach(track => track.stop());
                    video.srcObject = null;
                    stream = null;
                }}
                
                startBtn.style.display = 'inline-block';
                stopBtn.style.display = 'none';
                statusDiv.style.display = 'none';
            }}
            
            // รับข้อมูลจาก Streamlit
            window.addEventListener('message', function(event) {{
                if (event.data.type === 'detection_result') {{
                    const result = event.data.data;
                    updateStatus(result);
                }}
            }});
            
            function updateStatus(result) {{
                if (!result) return;
                
                let statusClass = 'status-good';
                let statusText = '✅ All wearing masks!';
                
                if (result.without_mask > 0) {{
                    statusClass = 'status-danger';
                    statusText = `⚠️ ${result.without_mask} person(s) without mask detected!`;
                }} else if (result.total_faces === 0) {{
                    statusClass = 'status-warning';
                    statusText = '👤 No faces detected';
                }}
                
                statusDiv.className = 'status ' + statusClass;
                statusDiv.innerHTML = statusText + `<br>👥 Total: ${result.total_faces} | ✅ With Mask: ${result.with_mask}`;
            }}
        </script>
    </body>
    </html>
    """
    
    return st.components.v1.html(html_code, height=700, scrolling=False)

# ===== PAGE: DETECTION =====
if page == "🏠 Detection":
    st.title("😷 Face Mask Detection System")
    st.markdown("---")
    
    # คำแนะนำการใช้งาน
    with st.expander("ℹ️ วิธีการใช้งาน", expanded=False):
        st.markdown("""
        ### 📖 คู่มือการใช้งาน
        
        #### 📷 **โหมดอัปโหลดรูป:**
        - อัปโหลดรูปภาพเพื่อตรวจจับ
        
        #### 🎥 **โหมด Real-time Detection:**
        - คลิก "Start Detection" เพื่อเปิดกล้อง
        - ระบบจะตรวจจับอัตโนมัติทุก 0.5 วินาที
        - ดูผลลัพธ์แบบเรียลไทม์ใต้วิดีโอ
        - **ใช้งานได้บน Mobile & Desktop!**
        
        #### ⚙️ **การตั้งค่า:**
        - ปรับ Detection Threshold ที่แถบด้านซ้าย
        
        #### 💡 **เคล็ดลับ:**
        - แสงสว่างดี = ตรวจจับแม่นยำขึ้น
        - หน้าตรงกล้อง = ตรวจจับง่ายขึ้น
        """)
    
    detection_method = st.radio(
        "Choose Detection Method:",
        ["📷 Upload Image", "🎥 Real-time Camera"],
        horizontal=True
    )
    
    # ===== Upload Image Mode =====
    if detection_method == "📷 Upload Image":
        st.header("📷 Upload Image for Detection")
        
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            with st.spinner('🔍 Detecting...'):
                result_image, results, num_faces = detect_mask(image.copy(), threshold)
            
            with col2:
                st.subheader("Detection Result")
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # บันทึกประวัติ
            timestamp = datetime.now()
            with_mask = sum(1 for r in results if r['has_mask'])
            without_mask = num_faces - with_mask
            
            st.session_state.detection_history.append({
                'timestamp': timestamp,
                'total_faces': num_faces,
                'with_mask': with_mask,
                'without_mask': without_mask,
                'method': 'Upload Image'
            })
            
            # แสดงสถิติ
            st.markdown("---")
            st.subheader("📊 Detection Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👥 Total Faces", num_faces)
            
            with col2:
                st.metric("✅ With Mask", with_mask)
            
            with col3:
                st.metric("❌ Without Mask", without_mask)
            
            with col4:
                if num_faces > 0:
                    compliance = (with_mask / num_faces) * 100
                    st.metric("📈 Compliance", f"{compliance:.1f}%")
            
            if without_mask > 0:
                st.error("⚠️ Warning: People without masks detected!")
            elif num_faces > 0:
                st.success("✅ All people are wearing masks!")
    
    # ===== Real-time Camera Mode =====
    else:
        st.header("🎥 Real-time Detection")
        
        st.info("""
        📱 **Works on Mobile (iOS/Android) & Desktop!**
        
        - Click "Start Detection" to begin
        - Detection updates every 0.5 seconds
        - Results shown below video
        """)
        
        # Real-time camera component
        camera_data = realtime_camera_component()
        
        # ประมวลผลข้อมูลจากกล้อง
        if camera_data:
            try:
                # Decode base64 image
                image_data = camera_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                image_array = np.array(image)
                
                # แปลงเป็น BGR
                if len(image_array.shape) == 3:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = image_array
                
                # ตรวจจับ
                _, results, num_faces = detect_mask(image_bgr, threshold)
                
                with_mask = sum(1 for r in results if r['has_mask'])
                without_mask = num_faces - with_mask
                
                # ส่งผลกลับไปยัง JavaScript
                st.write(f"""
                <script>
                    window.parent.postMessage({{
                        type: 'detection_result',
                        data: {{
                            total_faces: {num_faces},
                            with_mask: {with_mask},
                            without_mask: {without_mask}
                        }}
                    }}, '*');
                </script>
                """, unsafe_allow_html=True)
                
                # บันทึกประวัติทุก 10 frames
                st.session_state.frame_count += 1
                if st.session_state.frame_count % 10 == 0 and num_faces > 0:
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now(),
                        'total_faces': num_faces,
                        'with_mask': with_mask,
                        'without_mask': without_mask,
                        'method': 'Real-time Camera'
                    })
                
            except Exception as e:
                pass  # ข้ามข้อผิดพลาดเพื่อให้ stream ไหลต่อ

# ===== PAGE: REPORTS =====
else:
    st.title("📊 Detection Reports")
    st.markdown("---")
    
    if len(st.session_state.detection_history) == 0:
        st.info("📭 No detection data yet. Start detecting to see reports!")
    else:
        df = pd.DataFrame(st.session_state.detection_history)
        
        # Summary Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_detections = len(df)
            st.metric("🔍 Total Detections", total_detections)
        
        with col2:
            total_faces = df['total_faces'].sum()
            st.metric("👥 Total Faces", int(total_faces))
        
        with col3:
            total_with_mask = df['with_mask'].sum()
            st.metric("✅ With Mask", int(total_with_mask))
        
        with col4:
            total_without_mask = df['without_mask'].sum()
            st.metric("❌ Without Mask", int(total_without_mask))
        
        st.markdown("---")
        
        # กราฟ
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Compliance Over Time")
            df['compliance'] = (df['with_mask'] / df['total_faces'] * 100).fillna(0)
            fig = px.line(df, x='timestamp', y='compliance', 
                         title='Mask Compliance Rate (%)',
                         labels={'compliance': 'Compliance (%)', 'timestamp': 'Time'})
            fig.update_traces(line_color='#00cc96')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🥧 Overall Distribution")
            labels_pie = ['With Mask', 'Without Mask']
            values_pie = [total_with_mask, total_without_mask]
            colors_pie = ['#00cc96', '#ef553b']
            
            fig = go.Figure(data=[go.Pie(labels=labels_pie, values=values_pie, 
                                         marker_colors=colors_pie)])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ตารางข้อมูล
        st.subheader("📋 Detection History")
        
        df_display = df.copy()
        df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_display = df_display.rename(columns={
            'timestamp': 'Time',
            'total_faces': 'Total Faces',
            'with_mask': 'With Mask',
            'without_mask': 'Without Mask',
            'method': 'Method'
        })
        
        st.dataframe(df_display, use_column_width=True, height=400)
        
        # ปุ่มล้างข้อมูล
        st.markdown("---")
        if st.button("🗑️ Clear All History", type="secondary"):
            st.session_state.detection_history = []
            st.session_state.frame_count = 0
            st.rerun()