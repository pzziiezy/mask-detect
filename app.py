import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

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
        # ลอง VGG16 ก่อน
        model = keras.models.load_model('models/mask_detector_vgg16.h5')
        return model, 224  # VGG16 ใช้ 224x224
    except:
        try:
            # ถ้าไม่มี ใช้โมเดลเดิม
            model = keras.models.load_model('models/mask_detector.h5')
            return model, 128  # CNN ใช้ 128x128
        except Exception as e:
            st.error(f"Cannot load model: {e}")
            return None, None

model, IMG_SIZE = load_model()

if model is None:
    st.error("❌ ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบไฟล์โมเดล")
    st.stop()

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    st.error(f"Cannot load face detector: {e}")
    st.stop()

# ตรวจสอบว่าใช้โมเดลไหน
model_type = "VGG16" if IMG_SIZE == 224 else "Custom CNN"

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Detection", "📊 Reports"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.3, 0.05)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Model:** {model_type}\n\n"
    f"**Image Size:** {IMG_SIZE}x{IMG_SIZE}\n\n"
    "**Dataset:** 12K Images\n\n"
    "📱 **Mobile Compatible**"
)

# ฟังก์ชันตรวจจับ
def detect_mask(image, threshold=0.5):
    labels = ['❌ No Mask', '✅ With Mask']
    colors = [(255, 0, 0), (0, 255, 0)]
    
    # ตรวจสอบ input image
    if image is None or image.size == 0:
        return image, [], 0
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    except Exception as e:
        st.error(f"Face detection error: {e}")
        return image, [], 0
    
    results = []
    
    for (x, y, w, h) in faces:
        try:
            # ขยายกรอบหน้าเล็กน้อย
            padding = 20
            y1 = max(0, y - padding)
            y2 = min(image.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(image.shape[1], x + w + padding)
            
            face_img = image[y1:y2, x1:x2]
            
            if face_img.size == 0:
                continue
            
            # Resize
            face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            
            # Normalize - แก้ตรงนี้
            face_img = face_img.astype('float32') / 255.0
            
            # เพิ่มมิติ
            face_img = np.expand_dims(face_img, axis=0)
            
            # Predict
            prediction = model.predict(face_img, verbose=0)[0][0]
            label_idx = 1 if prediction > threshold else 0
            label = labels[label_idx]
            confidence = prediction if label_idx == 1 else (1 - prediction)
            color = colors[label_idx]
            
            # วาดกรอบ
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
            
        except Exception as e:
            # ข้ามใบหน้าที่มีปัญหา
            continue
    
    return image, results, len(faces)

# ===== PAGE: DETECTION =====
if page == "🏠 Detection":
    st.title("😷 Face Mask Detection System")
    st.markdown("---")
    
    # คำแนะนำการใช้งาน
    with st.expander("ℹ️ วิธีการใช้งาน", expanded=False):
        st.markdown(f"""
        ### 📖 คู่มือการใช้งาน
        
        **🤖 โมเดลที่ใช้:** {model_type}
        
        #### 📷 **โหมดอัปโหลดรูป:**
        - อัปโหลดรูปภาพเพื่อตรวจจับ
        - รองรับไฟล์: JPG, JPEG, PNG
        
        #### 📸 **โหมดถ่ายรูป (Mobile & Desktop):**
        - คลิก "Take a photo" เพื่อเปิดกล้อง
        - ถ่ายรูปแล้วรอผลลัพธ์
        - **ใช้งานได้บนมือถือ!**
        
        #### ⚙️ **การตั้งค่า:**
        - **Detection Threshold**: ปรับความแม่นยำ
          - ต่ำ (0.2-0.3) = ตรวจจับหน้ากากง่ายขึ้น
          - สูง (0.5-0.7) = ตรวจจับหน้ากากเข้มงวดขึ้น
        
        #### 💡 **เคล็ดลับ:**
        - แสงสว่างดี = ตรวจจับแม่นยำขึ้น
        - หน้าตรงกล้อง = ตรวจจับง่ายขึ้น
        - {model_type} ทำงานได้ดีกับหน้าเอียงและแสงน้อย
        """)
    
    detection_method = st.radio(
        "Choose Detection Method:",
        ["📷 Upload Image", "📸 Take Photo"],
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
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with st.spinner('🔍 Detecting...'):
                result_image, results, num_faces = detect_mask(image.copy(), threshold)
            
            with col2:
                st.subheader("Detection Result")
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
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
    
    # ===== Camera Mode =====
    else:
        st.header("📸 Camera Detection")
        st.info("📱 **Works on Mobile & Desktop!** Click the button below to take a photo")
        
        # Camera Component
        camera_input = st.camera_input("Take a photo", key="camera")
        
        if camera_input is not None:
            # อ่านรูปจากกล้อง
            image = Image.open(camera_input)
            image_array = np.array(image)
            
            # แปลงจาก RGB เป็น BGR สำหรับ OpenCV
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 4:  # RGBA
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                else:  # RGB
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Captured Image")
                st.image(image, use_container_width=True)
            
            with st.spinner('🔍 Detecting...'):
                result_image, results, num_faces = detect_mask(image_bgr.copy(), threshold)
            
            with col2:
                st.subheader("Detection Result")
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # บันทึกประวัติ
            timestamp = datetime.now()
            with_mask = sum(1 for r in results if r['has_mask'])
            without_mask = num_faces - with_mask
            
            st.session_state.detection_history.append({
                'timestamp': timestamp,
                'total_faces': num_faces,
                'with_mask': with_mask,
                'without_mask': without_mask,
                'method': 'Camera'
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
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # ปุ่มล้างข้อมูล
        st.markdown("---")
        if st.button("🗑️ Clear All History", type="secondary"):
            st.session_state.detection_history = []
            st.rerun()