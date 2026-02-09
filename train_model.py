# นำเข้า libraries ที่จำเป็น
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

print(" เริ่มต้นโปรเจค Mask Detection!")
print(f"TensorFlow version: {tf.__version__}")

# ตั้งค่าพื้นฐาน
IMG_SIZE = 128  # ขนาดรูปที่จะใช้ train
BATCH_SIZE = 32  # จำนวนรูปที่ประมวลผลพร้อมกัน
EPOCHS = 50  # จำนวนรอบในการ train

# เส้นทางไปยังข้อมูล
train_dir = 'data/train'
validation_dir = 'data/Validation'
test_dir = 'data/test'

print("\n📁 กำลังเตรียมข้อมูล...")

# เตรียมข้อมูลสำหรับ Training (มี Data Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # ปรับค่าสีให้อยู่ระหว่าง 0-1
    rotation_range=20,  # หันรูปได้ 20 องศา
    width_shift_range=0.2,  # เลื่อนรูปซ้าย-ขวา
    height_shift_range=0.2,  # เลื่อนรูปบน-ล่าง
    horizontal_flip=True,  # พลิกรูปซ้าย-ขวา
    zoom_range=0.2  # ซูมเข้า-ออก
)

# เตรียมข้อมูลสำหรับ Validation และ Test (ไม่มี Augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# โหลดข้อมูล
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # 2 คลาส: with_mask, without_mask
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\n✅ จำนวนรูป Training: {train_generator.samples}")
print(f"✅ จำนวนรูป Validation: {validation_generator.samples}")
print(f"✅ จำนวนรูป Test: {test_generator.samples}")
print(f"📊 Classes: {train_generator.class_indices}")

# สร้างโมเดล CNN
print("\n🏗️ กำลังสร้างโมเดล...")

model = keras.Sequential([
    # ชั้นที่ 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # ชั้นที่ 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # ชั้นที่ 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # ชั้นที่ 4
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # แปลงเป็น 1 มิติ
    layers.Flatten(),
    
    # Fully Connected Layers
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # ป้องกัน overfitting
    layers.Dense(1, activation='sigmoid')  # Output: 0 หรือ 1
])

# Compile โมเดล
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n📋 สรุปโครงสร้างโมเดล:")
model.summary()

# Train โมเดล
print(f"\n🎯 เริ่ม Training {EPOCHS} รอบ...")
print("⏳ อาจใช้เวลาสักครู่ โปรดรอ...\n")

# หยุดเทรนอัตโนมัติเมื่อ val_loss ไม่ดีขึ้นใน 5 รอบติดต่อกัน
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping],
    verbose=1
)

# บันทึกโมเดล
model_path = 'models/mask_detector.h5'
model.save(model_path)
print(f"\n💾 บันทึกโมเดลแล้วที่: {model_path}")

# ประเมินผลด้วย Test set
print("\n📊 กำลังประเมินผลด้วย Test set...")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# สร้างกราฟแสดงผล
print("\n📈 กำลังสร้างกราฟผลลัพธ์...")

plt.figure(figsize=(12, 4))

# กราฟ Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# กราฟ Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('models/training_history.png')
print("💾 บันทึกกราฟแล้วที่: models/training_history.png")

print("\n🎉 เสร็จสิ้น! โมเดลพร้อมใช้งานแล้ว")