import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import threading

# ==========================================
# 🧠 1. นิยามโครงสร้างโมเดล Action GRU (5 คลาส)
# ==========================================
class ActionGRU(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=5):
        super(ActionGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Timestep=30, Features=34)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :]) # เอาเฟรมสุดท้ายไปจัดหมวดหมู่
        return out

# ==========================================
# ⚙️ 2. ตั้งค่า Path โมเดลที่คุณเทรนเอง
# ==========================================
YOLO_WEIGHTS = "yolo11n-pose.pt"            # หรือใส่ Path โมเดล YOLO ที่คุณ Fine-tune เอง
GRU_WEIGHTS = "backend/model/action_gru_model.pt"         # ไฟล์น้ำหนัก GRU 5 คลาสที่เทรนเสร็จจากขั้นตอนก่อนหน้า

# นิยามชื่อคลาสภาษาไทยและสีที่จะแสดงบนหัวของแต่ละคน
CLASS_MAP = {
    0: {"name": "ปกติ (Normal)", "color": (0, 255, 0)},     # เขียว
    1: {"name": "💥 ต่อสู้ (Fight)", "color": (0, 0, 255)},    # แดง
    2: {"name": "🏃‍♂️ วิ่ง (Run)", "color": (255, 255, 0)},    # ฟ้า/เหลือง
    3: {"name": "🧎‍♂️ ล้มลง (Fall)", "color": (0, 165, 255)},  # ส้ม
    4: {"name": "🔫 ยิง/ใช้อาวุธ (Shoot)", "color": (255, 0, 255)} # ชมพู
}

WINDOW_SIZE = 30 # GRU ต้องการข้อมูลต่อเนื่อง 30 เฟรม

# ==========================================
# 🎬 3. ฟังก์ชันหลักของฝั่ง Vision (สำหรับเรียกเปิดกล้อง Real-time)
# ==========================================
def vision_realtime_thread(shared_results=None, data_lock=None):
    print("🎬 กำลังเริ่มต้นระบบ AI Vision...")
    
    # เช็กฮาร์ดแวร์สำหรับการ์ดจอ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Vision รันบนเดไวซ์: {device}")
    
    # โหลดโมเดล YOLO และ GRU ที่เป็นฝีมือการเทรนของคุณเอง
    yolo_model = YOLO(YOLO_WEIGHTS)
    
    gru_model = ActionGRU(input_size=34, hidden_size=64, num_layers=2, num_classes=5).to(device)
    gru_model.load_state_dict(torch.load(GRU_WEIGHTS, map_location=device))
    gru_model.eval() # เปิดโหมดทดสอบ
    
    # เปิดกล้องโน้ตบุ๊กแบบ Real-time (เลข 0 คือกล้องหลักของเครื่อง)
    cap = cv2.VideoCapture(0)
    
    # กระเป๋าความจำสำหรับทำหน้าต่างสะสมเฟรมแยกตามรายบุคคล (Track ID)
    # โครงสร้าง: { track_id: [[เฟรม1_34จุด], [เฟรม2_34จุด], ...] }
    buffers = {}
    
    print("📸 กล้องโน้ตบุ๊กเปิดใช้งานแล้ว! (กด 'q' เพื่อปิด)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ไม่สามารถดึงภาพจากกล้องโน้ตบุ๊กได้")
            break
            
        # 1. ส่งภาพเข้า YOLOv11 เพื่อแทร็กกิ้งคนและหาคีย์พอยต์แบบเรียลไทม์
        results = yolo_model.track(frame, persist=True, verbose=False)
        
        # คัดลอกเฟรมไว้สำหรับวาดตัวหนังสือแสดงผลเอง
        display_frame = frame.copy()
        
        current_frame_actions = [] # เอาไว้บันทึกว่าในเฟรมนี้เจอพฤติกรรมเด่นๆ อะไรบ้าง
        
        # ตรวจสอบว่าในเฟรมมีคนและแทร็ก ID ติดอยู่หรือไม่
        if len(results) > 0 and results[0].keypoints is not None and results[0].boxes.id is not None:
            result = results[0]
            
            # ดึงข้อมูลพิกัด บ็อกซ์ และไอดีของทุกคน
            keypoints_list = result.keypoints.xy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            
            h, w = frame.shape[:2]
            
            # 2. วนลูปประมวลผลท่าทางของ "ทุกคนในภาพพร้อมกัน"
            for i, track_id in enumerate(track_ids):
                person_kpts = keypoints_list[i].copy()
                bbox = boxes[i]
                
                # Normalize พิกัดคีย์พอยต์ให้มีค่าระหว่าง 0-1
                person_kpts[:, 0] /= w
                person_kpts[:, 1] /= h
                flat_pose = person_kpts.flatten() # ยุบเหลือเวกเตอร์ 34 มิติ
                
                # เช็กว่า ID นี้เคยเจอหรือยัง ถ้ายังให้สร้างคลังเฟรมว่างไว้
                if track_id not in buffers:
                    buffers[track_id] = []
                    
                # สะสมพิกัดเฟรมล่าสุดเข้าคลังของคนคนนั้น
                buffers[track_id].append(flat_pose)
                
                # จำกัดความยาวให้เก็บเฉพาะ 30 เฟรมล่าสุด (Sliding Window)
                if len(buffers[track_id]) > WINDOW_SIZE:
                    buffers[track_id].pop(0)
                    
                # กำหนดสถานะเริ่มต้นระหว่างสะสมเฟรม
                action_text = "กำลังคำนวณ..."
                text_color = (128, 128, 128)
                
                # 3. ส่งข้อมูลให้โมเดล GRU ประมวลผลเมื่อสะสมเฟรมของคนนั้นครบ 30 เฟรมต่อเนื่อง
                if len(buffers[track_id]) == WINDOW_SIZE:
                    input_tensor = torch.FloatTensor(np.array([buffers[track_id]])).to(device)
                    
                    with torch.no_grad():
                        outputs = gru_model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                        
                        # เลือกคลาสที่มีค่าเปอร์เซ็นต์ความมั่นใจสูงสุด
                        pred_class = np.argmax(probabilities)
                        confidence = probabilities[pred_class]
                        
                        # ตรวจจับเฉพาะกรณีที่ AI มั่นใจมากกว่า 60% เพื่อป้องกันหน้าจอเต้นกระพริบไปมา
                        if confidence > 0.60:
                            action_text = f"{CLASS_MAP[pred_class]['name']} ({confidence*100:.1f}%)"
                            text_color = CLASS_MAP[pred_class]['color']
                            current_frame_actions.append(CLASS_MAP[pred_class]['name'])
                
                # 4. วาดผลลัพธ์แยกตามบุคคลลงบนหน้าจอวิดีโอสด
                x1, y1, x2, y2 = map(int, bbox)
                # วาดกล่อง Bounding Box รอบตัวคน
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), text_color, 2)
                # เขียนชื่อ ID และคำทำนายจาก GRU ไว้บนหัวของคนนั้นๆ
                cv2.putText(display_frame, f"ID {track_id}: {action_text}", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
        
        # 5. ส่งค่าการทำนายฝั่งภาพขึ้นระบบแชร์กลาง (เพื่อเอาไป Fusion กับโมเดลเสียง)
        if shared_results is not None and data_lock is not None:
            # ดึงสถานะร้ายแรงที่สุดในเฟรมนั้นส่งออกไป (เช่น ถ้ามีคนสู้กันหรือยิงกัน ให้ส่งคลาสนั้นออกไปเลย)
            final_vision_status = "normal"
            for act in ["💥 ต่อสู้ (Fight)", "🔫 ยิง/ใช้อาวุธ (Shoot)", "🧎‍♂️ ล้มลง (Fall)", "🏃‍♂️ วิ่ง (Run)"]:
                if any(act in item for item in current_frame_actions):
                    final_vision_status = act.split()[-1] # ตัดคำเหลือแค่ชื่อคลาสสากล
                    break
            
            with data_lock:
                shared_results["vision_action"] = final_vision_status

        # แสดงหน้าต่างกล้อง Real-time พร้อมกราฟิก
        cv2.imshow("KU Arise Project - Realtime Vision (YOLO + GRU)", display_frame)
        
        # กดปุ่ม 'q' บนแป้นพิมพ์เพื่อปิดโปรแกรม
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("👋 ปิดกล้องฝั่ง Vision เรียบร้อยแล้ว")

# ==========================================
# 🚀 4. ทดสอบรันเฉพาะฝั่ง Vision แบบโดดๆ
# ==========================================
if __name__ == "__main__":
    # รันเดี่ยวๆ เพื่อทดสอบระบบภาพก่อนเอาไปประกบกับ Thread เสียง
    vision_realtime_thread()
