# นำเข้า libraries ที่จำเป็น
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

print("🚀 เริ่มต้นโปรเจค Mask Detection with VGG16!")
print(f"TensorFlow version: {tf.__version__}")

# ตั้งค่าพื้นฐาน
IMG_SIZE = 224  # VGG16 ใช้ 224x224
BATCH_SIZE = 32
EPOCHS = 30  # ลดลงเพราะ VGG16 เรียนเร็วกว่า

# เส้นทางไปยังข้อมูล
train_dir = 'data/train'
validation_dir = 'data/Validation'
test_dir = 'data/test'

print("\n📁 กำลังเตรียมข้อมูล...")

# Data Augmentation แบบเข้มข้น (เพิ่มความแข็งแกร่ง)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # เพิ่มจาก 20
    width_shift_range=0.3,  # เพิ่มจาก 0.2
    height_shift_range=0.3,
    horizontal_flip=True,
    zoom_range=0.3,  # เพิ่มจาก 0.2
    brightness_range=[0.5, 1.5],  # เพิ่มการปรับแสง
    shear_range=0.2,  # เพิ่มการบิด
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# โหลดข้อมูล
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
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

# สร้างโมเดล VGG16
print("\n🏗️ กำลังสร้างโมเดล VGG16...")

# โหลด VGG16 pre-trained (ไม่รวม top layer)
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model layers (ไม่ train ส่วนนี้)
base_model.trainable = False

# สร้างโมเดลใหม่
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile โมเดล
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n📋 สรุปโครงสร้างโมเดล:")
model.summary()

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)

# Train โมเดล (Phase 1: Frozen base)
print(f"\n🎯 Phase 1: Training with frozen VGG16 base...")
print(f"Training for {EPOCHS} epochs...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Fine-tuning (Phase 2: Unfreeze some layers)
print("\n🔥 Phase 2: Fine-tuning (unfreezing last 4 layers)...")

# Unfreeze ชั้นสุดท้ายของ VGG16
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Compile ใหม่ด้วย learning rate ต่ำกว่า
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train ต่อ
history_fine = model.fit(
    train_generator,
    epochs=10,  # Train ต่ออีก 10 epochs
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# บันทึกโมเดล
model_path = 'models/mask_detector_vgg16.h5'
model.save(model_path)
print(f"\n💾 บันทึกโมเดลแล้วที่: {model_path}")

# ประเมินผลด้วย Test set
print("\n📊 กำลังประเมินผลด้วย Test set...")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# รวม history จาก 2 phases
all_history = {
    'accuracy': history.history['accuracy'] + history_fine.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'] + history_fine.history['val_accuracy'],
    'loss': history.history['loss'] + history_fine.history['loss'],
    'val_loss': history.history['val_loss'] + history_fine.history['val_loss']
}

# สร้างกราฟแสดงผล
print("\n📈 กำลังสร้างกราฟผลลัพธ์...")

plt.figure(figsize=(14, 5))

# กราฟ Accuracy
plt.subplot(1, 2, 1)
plt.plot(all_history['accuracy'], label='Training Accuracy')
plt.plot(all_history['val_accuracy'], label='Validation Accuracy')
plt.axvline(x=len(history.history['accuracy']), color='r', linestyle='--', label='Fine-tuning starts')
plt.title('VGG16 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# กราฟ Loss
plt.subplot(1, 2, 2)
plt.plot(all_history['loss'], label='Training Loss')
plt.plot(all_history['val_loss'], label='Validation Loss')
plt.axvline(x=len(history.history['loss']), color='r', linestyle='--', label='Fine-tuning starts')
plt.title('VGG16 Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('models/training_history_vgg16.png')
print("💾 บันทึกกราฟแล้วที่: models/training_history_vgg16.png")

print("\n🎉 เสร็จสิ้น! โมเดล VGG16 พร้อมใช้งานแล้ว")
print(f"📊 Final Test Accuracy: {test_accuracy*100:.2f}%")