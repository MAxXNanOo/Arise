import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 📁 1. เตรียมเส้นทางโฟลเดอร์ Dataset
# ==========================================
# โครงสร้างโฟลเดอร์ปลายทาง:
# data/gru_dataset/normal/*.npy
# data/gru_dataset/fight/*.npy
# data/gru_dataset/run/*.npy
# data/gru_dataset/fall/*.npy
# data/gru_dataset/shoot/*.npy
DATA_DIR = "data/gru_dataset"
WINDOW_SIZE = 30  # จำนวนเฟรมคงที่ที่ต้องการป้อนให้ GRU (ล็อกไว้ที่ 30 เฟรม)
# ==========================================
# 🛠️ 2. สร้าง Custom Dataset สำหรับโหลดไฟล์ .npy (รองรับ 5 คลาส)
# ==========================================
class PoseSequenceDataset(Dataset):
    def __init__(self, base_dir, sequence_length=30):
        self.sequence_length = sequence_length
        self.samples = []
        self.labels = []
        
        # 🎯 แก้ไขคลาสเพิ่มเป็น 5 คลาสตรงนี้ เรียงลำดับจาก 0 ถึง 4
        self.class_mapping = {
            "normal": 0,
            "fight": 1,
            "run": 2,
            "fall": 3,
            "shoot": 4
        }
        
        for class_name, label_value in self.class_mapping.items():
            class_folder = os.path.join(base_dir, class_name)
            # ค้นหาไฟล์ .npy ทั้งหมดในโฟลเดอร์ของคลาสนั้นๆ
            file_paths = glob.glob(os.path.join(class_folder, "*.npy"))
            
            for path in file_paths:
                self.samples.append(path)
                self.labels.append(label_value)
                
        print(f"📊 โหลดข้อมูลเสร็จสิ้น: พบไฟล์ทั้งหมด {len(self.samples)} ไฟล์")
        # แสดงสรุปจำนวนไฟล์แต่ละคลาส
        for class_name, label_value in self.class_mapping.items():
            count = self.labels.count(label_value)
            print(f"   - คลาส [{class_name}] (Label {label_value}): {count} ไฟล์")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        label = self.labels[idx]
        
        # โหลดพิกัดคีย์พอยต์ Shape เดิม: (จำนวนเฟรม, 17, 2)
        sequence = np.load(file_path)
        
        # แปลงมิติจาก (Frames, 17, 2) ให้เป็นเส้นตรงยาว (Frames, 34) เพื่อเข้า GRU
        num_frames = sequence.shape[0]
        sequence = sequence.reshape(num_frames, -1)
        
        # --- กลไกจัดการความยาวเฟรมให้เท่ากับ WINDOW_SIZE ---
        if num_frames >= self.sequence_length:
            # กรณีที่เฟรมยาวเกิน: สุ่มตัดเอาเฉพาะช่วงที่ยาวเท่ากับ sequence_length
            start_idx = np.random.randint(0, num_frames - self.sequence_length + 1)
            sequence = sequence[start_idx : start_idx + self.sequence_length]
        else:
            # กรณีที่เฟรมสั้นเกิน: ทำ Padding โดยเติมค่า 0 ต่อท้ายให้ครบความยาว
            padding_length = self.sequence_length - num_frames
            padding = np.zeros((padding_length, sequence.shape[1]))
            sequence = np.vstack((sequence, padding))
            
        return torch.FloatTensor(sequence), torch.tensor(label, dtype=torch.long)

# ==========================================
# 🏗️ 3. นิยามโครงสร้างโมเดล Action GRU (รองรับ 5 คลาส)
# ==========================================
class ActionGRU(nn.Module):
    # 🎯 แก้ตัวแปร num_classes เริ่มต้นเป็น 5 คลาสตรงนี้
    def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=5):
        super(ActionGRU, self).__init__()
        # input_size = 34 (มาจาก 17 จุดพิกัด X, Y บีบแบนเป็นเส้นตรง)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes) # เลเยอร์สุดท้ายจ่ายผลลัพธ์ออก 5 คลาส
        
    def forward(self, x):
        # x shape: (Batch_Size, Timestep=30, Features=34)
        out, _ = self.gru(x)
        # ดึงเอาผลลัพธ์จากสถานะเฟรมสุดท้าย (-1) ไปทำการจำแนกประเภท (Classification)
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 🚀 4. ขั้นตอนการเตรียมระบบและการเทรน (Training Loop)
# ==========================================
if __name__ == "__main__":
    # เช็กการ์ดจอ (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ กำลังรันโมเดลบน: {device}")
    
    # ประกาศใช้งาน Dataset และ DataLoader
    dataset = PoseSequenceDataset(base_dir=DATA_DIR, sequence_length=WINDOW_SIZE)
    # หมายเหตุ: ในงานจริงควรแบ่งเป็น Train/Val Dataset ก่อนเข้า DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # ประกาศ Model (ส่งค่า num_classes=5 เข้าไป), Loss Function, และ Optimizer
    model = ActionGRU(input_size=34, hidden_size=64, num_layers=2, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 20
    print("\n🏋️ เริ่มต้นกระบวนการเทรนโมเดล GRU สำหรับ 5 คลาส...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # เคลียร์ค่าเกรเดียนต์
            optimizer.zero_grad()
            
            # Forward Pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward Pass และ อัปเดตน้ำหนักโมเดล
            loss.backward()
            optimizer.step()
            
            # คำนวณค่าสถิติ
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
            
        epoch_loss = running_loss / len(dataset)
        epoch_acc = (correct_preds / total_preds) * 100
        
        print(f"Epoch [{epoch+1}/{num_epochs}] -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
    # 💾 บันทึกโมเดลเก็บไว้ใช้งาน Real-time
    # torch.save(model.state_dict(), "action_gru_model.pt")
    torch.save(model.state_dict(), "backend/model/action_gru_model.pt")
    print("\n💾 บันทึกน้ำหนักโมเดลสำเร็จไปที่: action_gru_model.pt")
