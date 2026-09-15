import sys
import io
import csv
import numpy as np
import tensorflow_hub as hub
import pyaudio

# =====================================
# Settings
# =====================================
SAMPLE_RATE = 16000
CHUNK_SECONDS = 2                 # อัปเดตการทำนายผลทุกๆ 2 วินาที
TOP_N = 3

# จำนวนข้อมูลเสียงที่ต้องอ่านต่อหนึ่งรอบ (Sample Rate * จำนวนวินาที)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_SECONDS) 

def get_class_names(yamnet):
    class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
    class_names = []
    with tf_io_open(class_map_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            class_names.append(row[2])
    return class_names

def tf_io_open(path):
    import tensorflow as tf
    return io.TextIOWrapper(io.BytesIO(tf.io.read_file(path).numpy()))

def main():
    print("=" * 50)
    print("Loading YAMNet...")
    print("=" * 50)
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    class_names = get_class_names(yamnet)

    # คำหลักของเสียงปืนและระเบิดที่เราสนใจ
    gun_keywords = ["gun", "gunshot", "gunfire", "machine gun", "artillery", "explosion"]
    gun_indices = [
        i for i, name in enumerate(class_names)
        if any(k in name.lower() for k in gun_keywords)
    ]

    # =====================================
    # Initialize PyAudio
    # =====================================
    p = pyaudio.PyAudio()
    
    # เปิดสตรีมรับสัญญาณเสียงจากไมค์โน้ตบุ๊ก
    stream = p.open(
        format=pyaudio.paFloat32,       # ดึงค่าเสียงออกมาเป็น float32 (ตรงกับที่ YAMNet ต้องการ)
        channels=1,                     # รับเสียงแบบ Mono (1 ช่อง)
        rate=SAMPLE_RATE,               # Sample Rate 16kHz
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print("\n" + "=" * 50)
    print("🎤 MICROPHONE LIVE STREAMING STARTED...")
    print("Press Ctrl+C to stop listening.")
    print("=" * 50 + "\n")

    try:
        while True:
            # 1. อ่านข้อมูลเสียงจากไมโครโฟนแบบเรียลไทม์
            # exception_on_overflow=False ป้องกันโปรแกรมหลุดหากหน่วยความจำบัฟเฟอร์เต็ม
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            # 2. แปลงข้อมูลไบนารีเป็น numpy array (float32)
            audio_chunk = np.frombuffer(data, dtype=np.float32)

            # 3. Normalize ข้อมูลเสียงป้องกันสัญญาณเบาเกินไป
            peak = np.max(np.abs(audio_chunk))
            if peak > 0 and peak < 0.1:
                audio_chunk = audio_chunk / peak

            # 4. ส่งข้อมูลเสียงให้ YAMNet ทำนายผล
            scores, embeddings, spectrogram = yamnet(audio_chunk)
            scores_np = scores.numpy()

            # หาหมวดหมู่ที่คะแนนสูงสุด
            max_scores = np.max(scores_np, axis=0)
            top_indices = np.argsort(max_scores)[::-1][:TOP_N]

            # 5. แสดงผลบนหน้าจอคอนโซล
            print(f"\n📢 [LIVE STATUS] (Mic Peak Amplitude: {peak:.4f})")
            for i in top_indices:
                print(f"  🟢 {class_names[i]:30s}  {max_scores[i] * 100:.2f}%")

            # ตรวจสอบเสียงกลุ่มอันตราย (เสียงปืน/ระเบิด)
            print("  ⚠️ [Target Tracking: Gun/Explosion]")
            for i in gun_indices:
                rank = int(np.where(np.argsort(max_scores)[::-1] == i)[0][0]) + 1
                # ถ้าคะแนนเสียงกลุ่มปืนมากกว่า 10% ให้แสดงเป็นสีแดงเด่นขึ้นมา
                if max_scores[i] > 0.10:
                    print(f"    🚨 *ALERT* -> {class_names[i]:25s}  {max_scores[i] * 100:.2f}%  (rank {rank})")
                else:
                    print(f"    {class_names[i]:34s}  {max_scores[i] * 100:.2f}%  (rank {rank})")

    except KeyboardInterrupt:
        print("\nStopping Live Stream...")
    finally:
        # ปิดการทำงานของสตรีมเสียงอย่างปลอดภัย
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Live Stream Stopped Successfully.")

if __name__ == "__main__":
    main()
