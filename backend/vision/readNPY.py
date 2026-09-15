import numpy as np

# 1. โหลดไฟล์ข้อมูลขึ้นมา
data = np.load("vision/gru_dataset/sequence.npy")

# 2. ปริ้นดูโครงสร้างข้อมูลรวม (จะแสดงผลลัพธ์เป็น (995, 17, 2))
print("Data Shape:", data.shape)

# 3. ลองดึงข้อมูลพิกัด 17 จุด ของ "เฟรมแรกสุด" (Frame index 0) ออกมาดู
print("\n--- Keypoints of the First Frame (Frame 0) ---")
first_frame = data[0]
print(first_frame)

# 4. เจาะจงดูพิกัด [X, Y] ของจุดใดจุดหนึ่ง (เช่น จุดที่ 0 คือ จมูก)
print("\nNose coordinate in Frame 0 [X, Y]:", first_frame[0])
