import os
import pickle
import numpy as np
import librosa
import tensorflow_hub as hub
from pydub import AudioSegment  # ใช้ pydub แกะไฟล์ MP4 แทนลบรสาดั้งเดิม

# =====================================
# CONFIGURATION SETTINGS
# =====================================
RAW_DATASET_DIR = "data/audio"      # ปรับตามพาธโฟลเดอร์จริงของคุณ
WAV_DATASET_DIR = "data/audio_wav_ready" 
OUTPUT_MODEL_DIR = "audio_models"
OUTPUT_PICKLE_FILE = os.path.join(OUTPUT_MODEL_DIR, "yamnet_embeddings.pkl")

TARGET_SAMPLE_RATE = 16000 
SUPPORTED_EXTENSIONS = (
    '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma',
    '.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'
)

# =====================================
# PHASE 1: CONVERT ALL TO STANDARD .WAV
# =====================================
print("--- [PHASE 1] เริ่มต้นแปลงไฟล์ทุกประเภทให้เป็น .wav (16kHz, Mono) ---")

if not os.path.exists(RAW_DATASET_DIR):
    print(f"[Error] ไม่พบโฟลเดอร์ต้นทาง: {RAW_DATASET_DIR}")
    exit(1)

for class_name in os.listdir(RAW_DATASET_DIR):
    class_raw_path = os.path.join(RAW_DATASET_DIR, class_name)
    if not os.path.isdir(class_raw_path):
        continue
        
    class_wav_path = os.path.join(WAV_DATASET_DIR, class_name)
    os.makedirs(class_wav_path, exist_ok=True)
    
    print(f"\n📂 กำลังกวาดและแปลงไฟล์ในหมวดหมู่: [{class_name}]")
    
    for filename in os.listdir(class_raw_path):
        if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
            continue
            
        src_file_path = os.path.join(class_raw_path, filename)
        base_name = os.path.splitext(filename)[0]
        dst_file_path = os.path.join(class_wav_path, f"{base_name}.wav")
        
        if os.path.exists(dst_file_path):
            continue
            
        try:
            # 🌟 แก้ไข: ใช้ pydub + ffmpeg ดึงเสียงออกจากวิดีโอ/ไฟล์เสียงทุกรูปแบบ ปลอดภัย 100% บน Ubuntu
            sound = AudioSegment.from_file(src_file_path)
            
            # แปลงเป็น Mono และตั้งค่า Sample Rate เป็น 16000Hz ทันที
            sound = sound.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1)
            
            # ส่งออกเป็นไฟล์ .wav มาตรฐาน
            sound.export(dst_file_path, format="wav")
            print(f"   ✅ แปลงสำเร็จ: {filename} -> {base_name}.wav")
            
        except Exception as e:
            print(f"   ❌ ไม่สามารถแปลงไฟล์ {filename} ได้: {e}")

# =====================================
# PHASE 2: EXTRACT EMBEDDINGS TO .PKL
# =====================================
print("\n--- [PHASE 2] เริ่มต้นสกัด Feature จากไฟล์ .wav เข้าสู่ .pkl ---")
print("กำลังดาวน์โหลดและตั้งค่า YAMNet Model...")

# 🌟 แก้ไข: บรรทัดนี้ใส่ URL ตัวเต็มของโมเดลตามมาตรฐานของ Kaggle
# ห้ามใส่แค่ "https://kaggle.com" หรือ "https://tfhub.dev" โดดๆ เด็ดขาดครับ
YAMNET_MODEL_URL = "https://kaggle.com"

try:
    yamnet = hub.load(YAMNET_MODEL_URL)
    print("🟢 โหลดโมเดล YAMNet สำเร็จ!")
except Exception as e:
    print(f"🔴 เกิดข้อผิดพลาดในการดึงโมเดล: {e}")
    print("💡 คำแนะนำ: ลองล้างแคชเก่าด้วยคำสั่ง: rm -rf /tmp/tfhub_modules/*")
    exit(1)

X = []
y = []

# เดินทางไปอ่านจากโฟลเดอร์ WAV ที่เตรียมไว้ในเฟสแรก
if not os.path.exists(WAV_DATASET_DIR):
    print("🔴 ไม่พบโฟลเดอร์เสียงที่แปลงสภาพเสร็จสิ้น")
    exit(1)

for class_name in os.listdir(WAV_DATASET_DIR):
    class_path = os.path.join(WAV_DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
        
    print(f"\n🧠 สกัดคุณลักษณะหมวดหมู่: [{class_name}]")
    
    for filename in os.listdir(class_path):
        if not filename.lower().endswith('.wav'):
            continue
            
        file_path = os.path.join(class_path, filename)
        
        try:
            # โหลดไฟล์ .wav ที่มีความเสถียรสูงขึ้นมาก
            audio, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)
            audio = audio.astype(np.float32)
            
            # ทำ Normalize ปรับสมดุลความดัง
            max_value = np.max(np.abs(audio))
            if max_value > 0:
                audio /= max_value
                
            # ป้อนเข้า YAMNet
            scores, embeddings, spectrogram = yamnet(audio)
            
            # หาค่าเฉลี่ยคุณลักษณะเสียงทั้งไฟล์
            embedding_mean = np.mean(embeddings.numpy(), axis=0)
            
            X.append(embedding_mean)
            y.append(class_name)
            print(f"   ⭐ สกัดสำเร็จ -> {filename}")
        except Exception as e:
            print(f"   ❌ เกิดข้อผิดพลาดตอนประมวลผล {filename}: {e}")

# =====================================
# PACK & SAVE DATASET
# =====================================
if len(X) == 0:
    print("\n❌ ไม่มีข้อมูลเสียงใดที่สกัดสำเร็จ กรุณาเช็คความถูกต้องของไฟล์เสียงต้นทาง")
    exit(1)

X = np.array(X)
y = np.array(y)

print("\n" + "="*50)
print("📊 สรุปชุดข้อมูลสุดท้ายในไฟล์ .pkl:")
print(f"   X Shape (จำนวนตัวอย่างเสียง, ขนาดมิติ 1024): {X.shape}")
print(f"   y Shape (จำนวนป้ายชื่อคลาส): {y.shape}")
print("="*50)

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
with open(OUTPUT_PICKLE_FILE, "wb") as f:
    pickle.dump((X, y), f)

print(f"\n💾 บันทึกไฟล์ข้อมูลพร้อมใช้สำเร็จ! ที่พาธ: {OUTPUT_PICKLE_FILE}")
