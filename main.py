import threading
import time
import cv2
from ultralytics import YOLO

# ==========================================
# 📊 1. ส่วนกลางสำหรับแชร์ผลลัพธ์ (Shared State)
# ==========================================
# ใช้เก็บผลลัพธ์ล่าสุดของแต่ละโมเดล เพื่อให้ระบบหลักดึงไปแสดงผลหรือตัดสินใจ
shared_results = {
    "vision_action": "กำลังรอข้อมูลภาพ...",
    "audio_action": "กำลังรอข้อมูลเสียง...",
    "system_status": "ปกติ"
}

# Lock สำหรับป้องกันไม่ให้ Thread แย่งกันเขียนข้อมูลพร้อมกันจนพัง
data_lock = threading.Lock()

# ==========================================
# 🎬 Thread ที่ 1: ทำงานฝั่งภาพ (Vision Processing)
# ==========================================
def vision_thread_function():
    print("🎬 เริ่มต้นระบบ AI วิเคราะห์ภาพ...")
    model = YOLO("yolo11n-pose.pt") # หรือ weights/best.pt ของคุณ
    
    # เปิดกล้องหรือวิดีโอ (0 = เว็บแคม)
    cap = cv2.VideoCapture(0) 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # รัน YOLO (และอาจจะมี GRU มาประมวลผลต่อตรงนี้)
        results = model.track(frame, persist=True, verbose=False)
        
        # --- สมมติตัวอย่างผลลัพธ์ภาพที่ได้ ---
        detected_action = "normal" 
        if results and results[0].boxes.id is not None:
            # ตรงนี้คือจุดที่คุณเอาพิกัดส่งเข้าโมเดล GRU 5 คลาสเพื่อทำนายผล
            detected_action = "fight" # สมมติว่า GRU บอกว่าสู้กัน
            
        # อัปเดตผลลัพธ์ฝั่งภาพเข้าสู่ระบบส่วนกลาง
        with data_lock:
            shared_results["vision_action"] = detected_action
            
        # (ออพชันนัล) แสดงผลหน้าจอ
        annotated_frame = results[0].plot() if results else frame
        cv2.putText(annotated_frame, f"Vision: {detected_action}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("AI Vision Stream", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# 🎤 Thread ที่ 2: ทำงานฝั่งเสียง (Audio Processing)
# ==========================================
def audio_thread_function():
    print("🎤 เริ่มต้นระบบ AI วิเคราะห์เสียง...")
    # โหลดโมเดลเสียงของคุณตรงนี้ เช่น Audio Classification / Pytorch Model
    # model = LoadAudioModel()
    
    while True:
        # 💡 ตัวอย่าง Logic ฝั่งเสียง:
        # 1. ดึงข้อมูลจากไมโครโฟนมาช่วงเวลาสั้นๆ (เช่น ทุกๆ 1 วินาที)
        # 2. แปลงเป็น Spectrogram หรือ Feature เสียง
        # 3. ส่งเข้าโมเดลเสียงทำนายผล
        
        time.sleep(1) # สมมติว่าดึงเสียงมาประมวลผลทุกๆ 1 วินาที
        detected_audio = "normal_sound" 
        
        # สมมติโมเดลเสียงตรวจเจอเสียงปืน (shoot) หรือเสียงคนตะโกนสู้กัน
        # detected_audio = "gunshot" 
        
        # อัปเดตผลลัพธ์ฝั่งเสียงเข้าสู่ระบบส่วนกลาง
        with data_lock:
            shared_results["audio_action"] = detected_audio

# ==========================================
# 🧠 Thread ที่ 3: ตัวตัดสินใจส่วนกลาง (Decision Making)
# ==========================================
def decision_logic_thread():
    print("🧠 เริ่มต้นระบบวิเคราะห์และตัดสินใจส่วนกลาง...")
    while True:
        time.sleep(0.5) # เช็กสถานะรวมทุกๆ 0.5 วินาที
        
        with data_lock:
            v_act = shared_results["vision_action"]
            a_act = shared_results["audio_action"]
            
        # 🎯 ฟิวชันข้อมูล (Data Fusion) ร่วมกันระหว่างภาพและเสียง
        # เช่น ถ้าภาพเห็นคนปะทะกัน (fight) และเสียงมีเสียงกระแทกดุดันด้วย ความน่าจะเป็นจะสูงมาก
        if v_act == "fight" or a_act == "gunshot":
            print(f"🚨 [แจ้งเตือนอันตราย!] ตรวจพบเหตุร้าย -> ภาพ: {v_act} | เสียง: {a_act}")
            # ตรงนี้สามารถใส่คำสั่งให้แจ้งเตือนเข้า Line Notify / ล็อกข้อมูลลง Database ได้เลยครับ
        else:
            print(f"🟢 สถานะปัจจุบัน -> ภาพ: {v_act} | เสียง: {a_act}")

# ==========================================
# 🚀 4. ฟังก์ชันหลักสำหรับรันพร้อมกัน
# ==========================================
if __name__ == "__main__":
    # สร้าง Thread แยกกันทำงาน 3 ขาพร้อมกัน
    t_vision = threading.Thread(target=vision_thread_function, daemon=True)
    t_audio = threading.Thread(target=audio_thread_function, daemon=True)
    t_decision = threading.Thread(target=decision_logic_thread, daemon=True)
    
    # สั่งให้ทุกระบบเริ่มทำงานพร้อมกันในแบบ Background
    t_vision.start()
    t_audio.start()
    t_decision.start()
    
    # เลี้ยงโปรแกรมหลักไว้ไม่ให้ดับ จนกว่าจะกดปิดฝั่งหน้าต่างวิดีโอ
    while t_vision.is_alive():
        time.sleep(1)
        
    print("👋 ปิดระบบ AI ทั้งหมดเรียบร้อยแล้ว")
