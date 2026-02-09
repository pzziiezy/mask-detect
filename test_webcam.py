import cv2
import numpy as np
from tensorflow import keras
import os
import winsound
from datetime import datetime

print("🎥 เริ่มโปรแกรมตรวจจับหน้ากาก")

# โหลดโมเดล
model_path = 'models/mask_detector.h5'
if not os.path.exists(model_path):
    print(f"❌ ไม่พบโมเดลที่ {model_path}")
    exit()

model = keras.models.load_model(model_path)
print("✅ โหลดโมเดลสำเร็จ")

# โหลด Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ตั้งค่า
IMG_SIZE = 128
labels = ['Without Mask ⚠️', 'With Mask ✓']
colors = [(0, 0, 255), (0, 255, 0)]

# ตัวแปรสำหรับนับจำนวน
total_people_detected = 0
people_without_mask = 0
people_with_mask = 0

# ตัวแปรสำหรับติดตามใบหน้าที่นับแล้ว
tracked_faces = {}
face_id_counter = 0
tracking_threshold = 100  # ระยะที่ถือว่าเป็นคนเดียวกัน (pixels)

# ตัวแปรสำหรับควบคุมเสียง
last_beep_time = None
beep_cooldown = 2

# เปิด webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิด webcam ได้")
    exit()

print("✅ เปิด webcam สำเร็จ")
print("📹 กด 'q' เพื่อออกจากโปรแกรม")
print("📹 กด 'r' เพื่อรีเซ็ตตัวนับ")
print("🔊 จะมีเสียงแจ้งเตือนเมื่อตรวจพบคนไม่ใส่หน้ากาก\n")

def calculate_distance(pos1, pos2):
    """คำนวณระยะห่างระหว่าง 2 จุด"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def find_matching_face(current_pos, tracked_faces, threshold):
    """หาใบหน้าที่ตรงกันจากที่เคยติดตามไว้"""
    for face_id, data in tracked_faces.items():
        if calculate_distance(current_pos, data['last_pos']) < threshold:
            return face_id
    return None

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ ไม่สามารถอ่านภาพจาก webcam")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    has_no_mask = False
    current_frame_faces = []
    
    # วนลูปแต่ละใบหน้าที่พบ
    for (x, y, w, h) in faces:
        face_center = (x + w//2, y + h//2)
        
        # ครอปใบหน้าออกมา
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_img = face_img / 255.0
        # ปรับ contrast และ brightness
        face_img = cv2.convertScaleAbs(face_img, alpha=1.2, beta=10)
        face_img = np.expand_dims(face_img, axis=0)
        
        # ทำนาย
        prediction = model.predict(face_img, verbose=0)[0][0]
        label_idx = 1 if prediction > 0.3 else 0
        label = labels[label_idx]
        confidence = prediction if label_idx == 1 else (1 - prediction)
        color = colors[label_idx]
        
        # ตรวจสอบว่าเป็นใบหน้าที่เคยติดตามไว้หรือไม่
        matching_id = find_matching_face(face_center, tracked_faces, tracking_threshold)
        
        if matching_id is not None:
            # อัพเดทตำแหน่งของใบหน้าที่ติดตามอยู่
            tracked_faces[matching_id]['last_pos'] = face_center
            tracked_faces[matching_id]['frames_missing'] = 0
            face_id = matching_id
        else:
            # พบใบหน้าใหม่ - นับเพิ่ม
            face_id = face_id_counter
            face_id_counter += 1
            total_people_detected += 1
            
            if label_idx == 0:
                people_without_mask += 1
            else:
                people_with_mask += 1
            
            tracked_faces[face_id] = {
                'last_pos': face_center,
                'label': label_idx,
                'frames_missing': 0
            }
            
            print(f"👤 ตรวจพบคนใหม่ #{total_people_detected}: {label}")
        
        current_frame_faces.append(face_id)
        
        # ถ้าไม่ใส่หน้ากาก
        if label_idx == 0:
            has_no_mask = True
        
        # วาดกรอบและข้อความ
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        text = f"{label}: {confidence*100:.1f}%"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y-35), (x + text_width + 10, y), color, -1)
        cv2.putText(frame, text, (x+5, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # ลบใบหน้าที่หายไปนานเกินไป
    faces_to_remove = []
    for face_id in tracked_faces:
        if face_id not in current_frame_faces:
            tracked_faces[face_id]['frames_missing'] += 1
            if tracked_faces[face_id]['frames_missing'] > 30:  # หายไป 30 เฟรม
                faces_to_remove.append(face_id)
    
    for face_id in faces_to_remove:
        del tracked_faces[face_id]
    
    # เล่นเสียงแจ้งเตือน
    if has_no_mask:
        current_time = datetime.now()
        if last_beep_time is None or (current_time - last_beep_time).total_seconds() >= beep_cooldown:
            try:
                winsound.Beep(1000, 500)
                last_beep_time = current_time
            except:
                winsound.MessageBeep(winsound.MB_ICONHAND)
        
        warning_text = "WARNING: No Mask Detected!"
        cv2.putText(frame, warning_text, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # แสดงสถิติบนหน้าจอ
    stats_y = frame.shape[0] - 120
    
    # พื้นหลังสำหรับสถิติ
    cv2.rectangle(frame, (10, stats_y - 10), (400, frame.shape[0] - 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, stats_y - 10), (400, frame.shape[0] - 10), (255, 255, 255), 2)
    
    # ข้อความสถิติ
    cv2.putText(frame, f"Total People: {total_people_detected}", 
                (20, stats_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"With Mask: {people_with_mask}", 
                (20, stats_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Without Mask: {people_without_mask}", 
                (20, stats_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # คำนวณเปอร์เซ็นต์
    if total_people_detected > 0:
        compliance_rate = (people_with_mask / total_people_detected) * 100
        cv2.putText(frame, f"Compliance: {compliance_rate:.1f}%", 
                    (20, stats_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # แสดงผล
    cv2.imshow('Mask Detection - Press Q to quit, R to reset', frame)
    
    # จัดการ keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # รีเซ็ตตัวนับ
        total_people_detected = 0
        people_without_mask = 0
        people_with_mask = 0
        tracked_faces = {}
        face_id_counter = 0
        print("\n🔄 รีเซ็ตตัวนับเรียบร้อย\n")

# ปิดทุกอย่าง
cap.release()
cv2.destroyAllWindows()

# แสดงสรุปผล
print("\n" + "="*50)
print("📊 สรุปผลการตรวจจับ")
print("="*50)
print(f"👥 จำนวนคนทั้งหมด: {total_people_detected} คน")
print(f"✅ ใส่หน้ากาก: {people_with_mask} คน")
print(f"❌ ไม่ใส่หน้ากาก: {people_without_mask} คน")
if total_people_detected > 0:
    compliance = (people_with_mask / total_people_detected) * 100
    print(f"📈 อัตราการใส่หน้ากาก: {compliance:.1f}%")
print("="*50)
print("👋 ปิดโปรแกรมเรียบร้อย")