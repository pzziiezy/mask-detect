import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
import time
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="😷 Mask Detection",
    page_icon="😷",
    layout="wide"
)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

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
    "**Dataset:** 12K Images"
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

# ===== PAGE: DETECTION =====
if page == "🏠 Detection":
    st.title("😷 Face Mask Detection System")
    st.markdown("---")
    
    # คำแนะนำการใช้งาน
    with st.expander("ℹ️ วิธีการใช้งาน (คลิกเพื่อดู)", expanded=False):
        st.markdown("""
        ### 📖 คู่มือการใช้งาน
        
        #### 📷 **โหมดอัปโหลดรูป:**
        1. เลือก "📷 Upload Image"
        2. คลิก "Browse files" เพื่อเลือกรูปภาพ
        3. รองรับไฟล์: JPG, JPEG, PNG
        4. รอระบบตรวจจับ (1-2 วินาที)
        5. ดูผลลัพธ์ทางขวา พร้อมสถิติด้านล่าง
        
        #### 🎥 **โหมด Webcam แบบเรียลไทม์:**
        1. เลือก "🎥 Webcam (Real-time)"
        2. คลิกติ๊กถูก "Start Webcam"
        3. อนุญาตการเข้าถึงกล้องเมื่อเบราว์เซอร์ถาม
        4. ระบบจะตรวจจับอัตโนมัติ
        5. ดูสถิติแบบเรียลไทม์ด้านล่างกล้อง
        
        #### ⚙️ **การตั้งค่า:**
        - **Detection Threshold** (แถบเลื่อน): 
          - ต่ำ (0.2-0.3) = ตรวจจับหน้ากากง่ายขึ้น
          - สูง (0.5-0.7) = ตรวจจับหน้ากากเข้มงวดขึ้น
        
        #### 📊 **ดูรายงาน:**
        - คลิก "📊 Reports" ที่แถบด้านซ้าย
        - ดูกราฟและสถิติการตรวจจับทั้งหมด
        - คลิก "🗑️ Clear All History" เพื่อล้างข้อมูล
        
        #### 💡 **เคล็ดลับ:**
        - แสงสว่างดี = ตรวจจับแม่นยำขึ้น
        - หน้าตรงกล้อง = ตรวจจับง่ายขึ้น
        - หน้ากากสีเข้ม = อาจต้องปรับ Threshold
        """)
    
    detection_method = st.radio(
        "Choose Detection Method:",
        ["📷 Upload Image", "🎥 Webcam (Real-time)"],
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
    
    # ===== Webcam Mode =====
    else:
        st.header("🎥 Real-time Webcam Detection")
        st.warning("⚠️ Webcam mode is not supported on Streamlit Cloud. Please run locally using: `streamlit run app.py`")
        
        st.info("""
        **To use webcam detection:**
        1. Download this code
        2. Run locally: `streamlit run app.py`
        3. Your browser will ask for camera permission
        """)

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
        
        # กราฟแท่ง
        st.subheader("📊 Detection Breakdown")
        
        df_melted = df[['timestamp', 'with_mask', 'without_mask']].melt(
            id_vars='timestamp', 
            value_vars=['with_mask', 'without_mask'],
            var_name='Status', 
            value_name='Count'
        )
        df_melted['Status'] = df_melted['Status'].map({
            'with_mask': 'With Mask',
            'without_mask': 'Without Mask'
        })
        
        fig = px.bar(df_melted, x='timestamp', y='Count', color='Status',
                     color_discrete_map={'With Mask': '#00cc96', 'Without Mask': '#ef553b'},
                     title='Mask Detection Over Time')
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
            st.rerun()