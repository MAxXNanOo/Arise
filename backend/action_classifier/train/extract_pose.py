from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob  # เพิ่มโมดูลสำหรับสแกนหาไฟล์ในโฟลเดอร์

MODEL_PATH = "yolo11n-pose.pt"
VIDEO_DIR = "data/pose/test"      # 📁 เปลี่ยนเป็นโฟลเดอร์ที่เก็บวิดีโอทั้งหมด
OUTPUT_DIR = "data/gru_dataset"   # โฟลเดอร์สำหรับเก็บไฟล์แยกรายคน

model = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. ค้นหาไฟล์วิดีโอทั้งหมดในโฟลเดอร์ (รองรับทั้ง .mp4 และ .avi)
video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) + glob.glob(os.path.join(VIDEO_DIR, "*.avi"))

print(f"🔄 พบไฟล์วิดีโอทั้งหมด {len(video_files)} ไฟล์ ในโฟลเดอร์ '{VIDEO_DIR}'")

# 2. วนลูปประมวลผลทีละไฟล์วิดีโออัตโนมัติ
for video_path in video_files:
    # ดึงชื่อไฟล์วิดีโอหลักออกมา (เช่น sample_1)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n🎬 กำลังประมวลผลไฟล์: {video_name}...")
    
    cap = cv2.VideoCapture(video_path)
    
    # ใช้ Dictionary เก็บลำดับคีย์พอยต์แยกตาม ID ของแต่ละคนเฉพาะในคลิปนี้
    track_sequences = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ใช้ .track และตั้งค่า persist=True เพื่อติดตาม ID
        results = model.track(frame, persist=True, verbose=False)

        if len(results) == 0:
            continue

        result = results[0]

        # ตรวจสอบว่าเฟรมนี้มีทั่ง Keypoints และ Track ID หรือไม่
        if result.keypoints is None or result.boxes.id is None:
            continue

        # ดึงพิกัดคีย์พอยต์ และ รายชื่อ ID ของทุกคนในเฟรมนี้ออกมา
        keypoints_list = result.keypoints.xy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()

        h, w = frame.shape[:2]

        # วนลูปประมวลผล "ทุกคน" ที่ตรวจเจอในเฟรมปัจจุบันพร้อมกัน
        for i, track_id in enumerate(track_ids):
            person_kpts = keypoints_list[i].copy()

            # ถ้าคนนี้ยังไม่มีพื้นที่เก็บข้อมูล ให้สร้างลิสต์ว่างรอไว้
            if track_id not in track_sequences:
                track_sequences[track_id] = []

            # Normalize พิกัด X, Y ของคนนี้
            person_kpts[:, 0] /= w
            person_kpts[:, 1] /= h

            # บันทึกท่าทางของคนนี้ลงในกระเป๋าข้อมูลของ ID ตัวเอง
            track_sequences[track_id].append(person_kpts)

    cap.release()

    # 3. บันทึกไฟล์แยกรายบุคคลหลังจากอ่านวิดีโอนี้จบ ก่อนไปคลิปถัดไป
    if len(track_sequences) > 0:
        for track_id, sequence in track_sequences.items():
            # แปลงข้อมูลของแต่ละ ID เป็น numpy array
            sequence_np = np.array(sequence, dtype=np.float32)
            
            # ตั้งชื่อไฟล์ผลลัพธ์โดยมี ชื่อวิดีโอเดิม + ID ต่อท้าย เช่น sample_1_person_1.npy
            file_name = f"{video_name}_person_{track_id}.npy"
            output_path = os.path.join(OUTPUT_DIR, file_name)
            
            np.save(output_path, sequence_np)
            print(f"   ✅ บันทึกสำเร็จ: {file_name} | Shape: {sequence_np.shape}")
    else:
        print(f"   ⚠️ ไม่พบข้อมูลบุคคลที่สามารถ Track ได้ในไฟล์ {video_name}")

print("\n🎉 ประมวลผลและสร้าง Dataset เสร็จสิ้นครบทุกไฟล์แล้ว!")




















# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os
# import torch
# import torch.nn as nn

# # ==========================================
# # 🧠 1. นิยามโครงสร้างโมเดล GRU สำหรับทำนายท่าทาง
# # ==========================================
# class ActionGRU(nn.Module):
#     def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=3):
#         super(ActionGRU, self).__init__()
#         # 34 มาจาก 17 จุดคีย์พอยต์ x พิกัด (X, Y)
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
        
#     def forward(self, x):
#         # x shape: (Batch, Timestep=30, Features=34)
#         out, _ = self.gru(x)
#         # เอาเฉพาะข้อมูลผลลัพธ์ของเฟรมสุดท้าย (Last Time Step) มาเข้า Fully Connected
#         out = self.fc(out[:, -1, :])
#         return out

# MODEL_PATH = "yolo11n-pose.pt"
# VIDEO_PATH = "data/pose/test/sample_1.mp4"
# # OUTPUT_PATH = "vision/gru_dataset/sequence.npy"
# OUTPUT_PATH = "data/gru_dataset/sequence.npy"

# # เตรียมโมเดล YOLO และ GRU
# model = YOLO(MODEL_PATH)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# gru_model = ActionGRU(input_size=34, hidden_size=64, num_layers=2, num_classes=3).to(device)
# gru_model.eval() # ตั้งโหมด Evaluation

# # รายชื่อคลาสพฤติกรรมที่จะให้ GRU บอก
# CLASS_NAMES = {
#     0: "กำลังยืน หรือ เดิน (Standing/Walking)",
#     1: "กำลังนั่ง หรือ หมอบ (Sitting/Squatting)",
#     2: "💥 กำลังต่อยกัน / ปะทะ (Fighting/Punching)"
# }

# cap = cv2.VideoCapture(VIDEO_PATH)

# sequence = []
# actions_log = [] 
# window_buffer = [] # หน้าต่างสะสมเฟรมสำหรับส่งให้ GRU (ต้องการ 30 เฟรมต่อเนื่อง)
# WINDOW_SIZE = 30

# print("🎬 กำลังเริ่มวิเคราะห์พฤเจกรรมด้วย YOLO + GRU...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame, verbose=False)

#     if len(results) == 0:
#         continue

#     result = results[0]

#     if result.keypoints is None or len(result.keypoints.xy) == 0:
#         continue

#     # เอาคนแรก
#     person = result.keypoints.xy[0].cpu().numpy()

#     # Normalize เก็บลง Dataset
#     h, w = frame.shape[:2]
#     person_norm = person.copy()
#     person_norm[:, 0] /= w
#     person_norm[:, 1] /= h
#     sequence.append(person_norm)

#     # แปลงพิกัด (17, 2) ให้เป็นเส้นตรงยาว (34,) เพื่อป้อนเข้า GRU
#     flat_pose = person_norm.flatten()
#     window_buffer.append(flat_pose)

#     # กำหนดค่าเริ่มต้นเผื่อกรณีเฟรมแรกๆ ยังสะสมข้อมูลไม่ครบ
#     action = "กำลังรวบรวมเฟรมเพื่อส่งให้ GRU..."

#     # --- 🧠 ส่งข้อมูลให้ GRU ประมวลผลเมื่อเฟรมต่อเนื่องครบกำหนด ---
#     if len(window_buffer) == WINDOW_SIZE:
#         try:
#             # แปลงเป็น Tensor ขนาด (Batch=1, Timestep=30, Features=34)
#             input_tensor = torch.FloatTensor(np.array([window_buffer])).to(device)
            
#             with torch.no_grad():
#                 outputs = gru_model(input_tensor)
#                 probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
#                 # หาคลาสที่โมเดลให้ค่าน้ำหนักมากที่สุด
#                 pred_class = np.argmax(probabilities)
#                 confidence = probabilities[pred_class] * 100
                
#                 action = f"{CLASS_NAMES[pred_class]} ({confidence:.1f}%)"
            
#             actions_log.append(CLASS_NAMES[pred_class])
            
#         except Exception as e:
#             action = "เกิดข้อผิดพลาดในระบบ GRU"

#         # เลื่อนหน้าต่างเวลา (Slide Window) โดยเอาเฟรมเก่าสุดออกเพื่อรับเฟรมใหม่ในลูปถัดไป
#         window_buffer.pop(0)

#     # --- 📺 แสดงผลลัพธ์บนภาพวิดีโอสดๆ ---
#     frame = result.plot()
    
#     cv2.putText(frame, f"GRU Detect: {action}", (30, 50), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
#     cv2.imshow("Real-time GRU Classification", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # แปลงข้อมูลและบันทึกคลังไฟล์ตามเดิม
# sequence = np.array(sequence, dtype=np.float32)
# os.makedirs("vision/gru_dataset", exist_ok=True)
# np.save(OUTPUT_PATH, sequence)

# # --- 📊 สรุปผลลัพธ์พฤติกรรมจาก GRU ---
# print("\n" + "="*40)
# print("📊 สรุปพฤติกรรมจากโมเดล GRU ในคลิปนี้")
# print("="*40)
# if actions_log: 
#     unique_actions, counts = np.unique(actions_log, return_counts=True)
#     for act, cnt in zip(unique_actions, counts):
#         percentage = (cnt / len(actions_log)) * 100
#         print(f"- {act}: เจอทั้งหมด {cnt} เฟรม ({percentage:.2f}%)")
# else:
#     print("เฟรมวิดีโอสั้นเกินกว่าที่ GRU จะประมวลผลได้ (ไม่ครบ 30 เฟรม)")
# print("="*40)

# print(f"Saved: {OUTPUT_PATH}")
# print(f"Shape: {sequence.shape}")
