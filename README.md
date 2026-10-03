# Arise (เเจ้งเตือนเมื่อพบเหตุการณ์ความไม่สงบ)
ทำเพื่อ

## Features
- เเจ้งเตือนเมื่อมีเหตุการณ์ คนต่อยตี คนก่อการร้าย เเละ เหตุการความวุ่นวาย

## 🛠เทคโนโลยีที่ใช้ (Tech Stack)
* **Frontend:** html, Tailwind 
* **Backend:** Python
* **AI:** YOLOv11n-pose, GRU, YOLOv11n, YAMNet

## 💻 วิธีการติดตั้งและเริ่มใช้งาน (Getting Started)

### 1. Prerequisites (สิ่งที่ต้องมีในเครื่องก่อน)
- Python


### 2. การติดตั้ง (Installation)
สร้าง .env ลงเครื่อง:
\`\`\`bash
python -m venv .env
download ;adlkfje
\`\`\`


### 3. เริ่มใช้งาน (Running the app)
รันโปรเจกต์ โดยใช้ .env:
\`\`\`bash
source .env/bin/activate
\`\`\`





## 📁 โครงสร้างและหน้าที่ของแต่ละไฟล์ (Project Structure & File Explanations)

```text
├── backend/
│   ├── action_classifier
│   │   ├── train/
|   |   |   ├── 
|   |   └──
│   ├── controllers/
│   │   └── userController.js
│   ├── models/
│   │   └── userModel.js
│   └── app.js
├── main.py
├── api_server.py
├── index.html
├── script.js
├── configAI.json
├── camera.json
└── README.md
```

### 📄 รายละเอียดและฟังก์ชันสำคัญในแต่ละไฟล์

#### 🔹 `backend/action_classifier`
ไฟล์หลักสำหรับเริ่มต้นระบบ (Entry Point) ทำหน้าที่ตั้งค่าเซิร์ฟเวอร์และเชื่อมต่อมิดเดิลแวร์
* `app.listen()` - สั่งให้เซิร์ฟเวอร์ทำงานตามพอร์ตที่กำหนด
* `errorHandler()` - ฟังก์ชันจัดการข้อผิดพลาด (Error Handling) ของระบบทั้งหมด

#### 🔹 `main.py`
ไฟล์การทำงานหลักของระบบไว้เริ่มต้นเเละคำนวนข้อมูล
* `EventCorrelationEngine` (Class) - 

#### 🔹 `src/controllers/userController.js`
ไฟล์ควบคุมลอจิกของระบบ (Controller) ที่เกี่ยวกับผู้ใช้งานทั้งหมด
* `registerUser(req, res)` - รับข้อมูล ตรวจสอบความถูกต้อง และบันทึกผู้ใช้ใหม่ลงฐานข้อมูล
* `loginUser(req, res)` - ตรวจสอบรหัสผ่านและสร้าง JWT Token เพื่อใช้ระบุตัวตน
* `getUserProfile(req, res)` - ดึงข้อมูลส่วนตัวของผู้ใช้ที่เข้าสู่ระบบอยู่

#### 🔹 `src/models/userModel.js`
ไฟล์กำหนดโครงสร้างข้อมูล (Schema) ของผู้ใช้งานในฐานข้อมูล
* `userSchema` - โครงสร้างที่กำหนดว่าข้อมูลผู้ใช้ต้องมี `name`, `email` (Unique), และ `password`
* `matchPassword(enteredPassword)` - ฟังก์ชันเปรียบเทียบรหัสผ่านที่ผู้ใช้พิมพ์เข้ามากับรหัสที่แฮชไว้ในระบบ

