import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# =====================================
# Load Models
# =====================================
print("=" * 50)
print("Loading YOLOv11 Pose...")
print("=" * 50)

yolo = YOLO("yolo11n-pose.pt")
print("[YOLO] Loaded successfully")

# =====================================
# Load GRU
# =====================================
print("\n" + "=" * 50)
print("Loading GRU model...")
print("=" * 50)

gru = tf.keras.models.load_model("models/gru_action.keras")
print("[GRU] Loaded successfully")

# =====================================
# Load Classes
# =====================================
classes = np.load("models/gru_classes.npy", allow_pickle=True)
print("[GRU] Classes:")
for i, name in enumerate(classes):
    print(f"    [{i}] {name}")

# =====================================
# Camera
# =====================================
print("\n" + "=" * 50)
print("Opening Camera...")
print("=" * 50)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit()

print("[CAMERA] Camera opened successfully")

# =====================================
# Sequence Configuration
# =====================================
sequence = []
frame_count = 0

# =====================================
# Main Loop
# =====================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read frame")
        break

    frame_count += 1
    print("\n" + "=" * 50)
    print(f"FRAME {frame_count}")
    print("=" * 50)

    # =================================
    # YOLO Detection
    # =================================
    print("[YOLO] Detecting Pose...")
    results = yolo(frame, verbose=False)

    # ป้องกันกรณี YOLO คืนค่า List ว่าง
    if not results:
        print("[YOLO] No results")
        cv2.imshow("YOLOv11 + GRU", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    result = results[0]

    # เช็คว่าเจอ Keypoints หรือไม่ (ป้องกัน Tensor ว่าง)
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        print("[YOLO] No person / pose detected")
        # ล้าง sequence ทิ้งหากไม่เจอคนต่อเนื่อง เพื่อไม่ให้ท่าทางเก่าตกค้าง (Optional)
        # sequence.clear() 
        cv2.imshow("YOLOv11 + GRU", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # แปลงเป็น NumPy Array ตั้งแต่แรก
    keypoints = result.keypoints.xy.cpu().numpy()
    num_detection = len(keypoints)
    print(f"[YOLO] Persons detected: {num_detection}")

    # ดึงข้อมูลคนแรก
    person = keypoints[0]  # shape: (17, 2)
    h, w = frame.shape[:2]

    # Copy เพื่อป้องกัน Warning และทำการ Normalize (x/w , y/h)
    person_norm = person.copy()
    person_norm[:, 0] /= w
    person_norm[:, 1] /= h

    # แก้ไขจุดสำคัญ: Flatten จาก (17, 2) ให้เป็น (34,) ก่อนเข้า Sequence
    person_flat = person_norm.flatten()

    # เพิ่มข้อมูลเข้า Sequence
    sequence.append(person_flat)
    print(f"[SEQUENCE] Current frames: {len(sequence)}/30")

    # รักษาความยาว Sequence ให้เท่ากับ 30 เฟรม
    if len(sequence) > 30:
        sequence.pop(0)

    # =================================
    # GRU Prediction
    # =================================
    if len(sequence) == 30:
        print("\n" + "-" * 50)
        print("[GRU] 30 frames collected")
        print("-" * 50)

        # แปลงเป็น มิติ (1, 30, 34) เพื่อส่งให้ GRU
        data = np.array(sequence, dtype=np.float32)
        data = np.expand_dims(data, axis=0)  # ผลลัพธ์จะได้ shape (1, 30, 34)

        print(f"[GRU] Input shape: {data.shape}")
        print("[GRU] Predicting...")

        prediction = gru.predict(data, verbose=0)[0]

        # แสดง Score ทุก Class
        for i, score in enumerate(prediction):
            print(f"    {classes[i]} : {score * 100:.2f}%")

        # หา Class ที่มั่นใจที่สุด
        index = np.argmax(prediction)
        confidence = prediction[index]
        label = classes[index]

        print("\n" + "-" * 50)
        print(f"[GRU] RESULT: {label}")
        print(f"[GRU] CONFIDENCE: {confidence * 100:.2f}%")
        print("-" * 50)

        # วาด Text ลงบนหน้าจอ
        text = f"{label}: {confidence * 100:.1f}%"
        cv2.putText(
            frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    # แสดงผลหน้าต่างกล้อง
    cv2.imshow("YOLOv11 + GRU", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n[PROGRAM] Quit")
        break

# =====================================
# Release Resources
# =====================================
cap.release()
cv2.destroyAllWindows()
print("\n[PROGRAM] Camera released")
print("[PROGRAM] Program finished")














































from ultralytics import YOLO

import tensorflow as tf

import cv2
import numpy as np

# =====================================

# Load Models

# =====================================

print("=" * 50)
print("Loading YOLOv11 Pose...")
print("=" * 50)

yolo = YOLO(
# "runs/pose/yolo11_pose/weights/best.pt"
"yolo11n-pose.pt"
)

print("[YOLO] Loaded successfully")

# =====================================

# Load GRU

# =====================================

print("\n" + "=" * 50)
print("Loading GRU model...")
print("=" * 50)

gru = tf.keras.models.load_model(
"models/gru_action.keras"
)

print("[GRU] Loaded successfully")

# =====================================

# Load Classes

# =====================================

classes = np.load(
"models/gru_classes.npy",
allow_pickle=True
)

print("[GRU] Classes:")

for i, name in enumerate(classes):
print(f"    [{i}] {name}")

# =====================================

# Camera

# =====================================

print("\n" + "=" * 50)
print("Opening Camera...")
print("=" * 50)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
print("[ERROR] Cannot open camera")
exit()

print("[CAMERA] Camera opened successfully")

# =====================================

# Sequence

# =====================================

sequence = []

frame_count = 0

# =====================================

# Main Loop

# =====================================

while True:

```
ret, frame = cap.read()

if not ret:
    print("[ERROR] Cannot read frame")
    break

frame_count += 1


print("\n" + "=" * 50)
print(f"FRAME {frame_count}")
print("=" * 50)


# =================================
# YOLO Detection
# =================================

print("[YOLO] Detecting Pose...")

results = yolo(
    frame,
    verbose=False
)


# =================================
# Check YOLO Result
# =================================

if len(results) == 0:

    print("[YOLO] No results")

    cv2.imshow(
        "YOLOv11 + GRU",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n[PROGRAM] Quit")
        break

    continue


result = results[0]


# =================================
# Check Keypoints
# =================================

if result.keypoints is None:

    print("[YOLO] No keypoints")

    cv2.imshow(
        "YOLOv11 + GRU",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n[PROGRAM] Quit")
        break

    continue


keypoints = result.keypoints.xy


# จำนวน Detection
num_detection = len(keypoints)

print(
    f"[YOLO] Persons detected: "
    f"{num_detection}"
)


if num_detection == 0:

    print(
        "[YOLO] No person / pose detected"
    )

else:

    # =================================
    # Get first person keypoints
    # =================================

    person = (
        keypoints[0]
        .cpu()
        .numpy()
    )


    print(
        f"[POSE] Keypoints shape: "
        f"{person.shape}"
    )

    print(
        f"[POSE] Number of keypoints: "
        f"{len(person)}"
    )


    # =================================
    # Frame size
    # =================================

    h, w = frame.shape[:2]

    print(
        f"[FRAME] Width: {w}"
    )

    print(
        f"[FRAME] Height: {h}"
    )


    # =================================
    # Normalize Keypoints
    # =================================

    person[:, 0] /= w
    person[:, 1] /= h


    print(
        "[POSE] Normalized Keypoints:"
    )

    print(person)


    # =================================
    # Show first keypoint
    # =================================

    print(
        "[POSE] First Keypoint (Nose):"
    )

    print(
        person[0]
    )


    # =================================
    # Add to Sequence
    # =================================

    sequence.append(
        person
    )


    print(
        f"[SEQUENCE] Current frames: "
        f"{len(sequence)}/30"
    )


# =================================
# Keep last 30 frames
# =================================

if len(sequence) > 30:

    sequence.pop(0)

    print(
        "[SEQUENCE] Removed oldest frame"
    )


# =================================
# GRU Prediction
# =================================

if len(sequence) == 30:

    print("\n" + "-" * 50)
    print("[GRU] 30 frames collected")
    print("-" * 50)


    # =================================
    # Convert to NumPy
    # =================================

    data = np.array(
        sequence,
        dtype=np.float32
    )


    print(
        f"[GRU] Before reshape: "
        f"{data.shape}"
    )


    # =================================
    # Reshape
    # =================================

    data = data.reshape(
        1,
        30,
        34
    )


    print(
        f"[GRU] Input shape: "
        f"{data.shape}"
    )


    # =================================
    # GRU Predict
    # =================================

    print("[GRU] Predicting...")

    prediction = gru.predict(
        data,
        verbose=0
    )[0]


    print(
        "[GRU] Prediction values:"
    )


    # =================================
    # Show every class score
    # =================================

    for i, score in enumerate(prediction):

        print(
            f"    {classes[i]} : "
            f"{score * 100:.2f}%"
        )


    # =================================
    # Get highest prediction
    # =================================

    index = np.argmax(
        prediction
    )


    confidence = prediction[index]

    label = classes[index]


    print("\n" + "-" * 50)

    print(
        f"[GRU] RESULT: {label}"
    )

    print(
        f"[GRU] CONFIDENCE: "
        f"{confidence * 100:.2f}%"
    )

    print("-" * 50)


    # =================================
    # Display text
    # =================================

    text = (
        f"{label}: "
        f"{confidence * 100:.1f}%"
    )


    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


# =================================
# Show Camera
# =================================

cv2.imshow(
    "YOLOv11 + GRU",
    frame
)


# =================================
# Quit
# =================================

if cv2.waitKey(1) & 0xFF == ord("q"):

    print("\n[PROGRAM] Quit")

    break
```

# =====================================

# Release

# =====================================

cap.release()

cv2.destroyAllWindows()

print("\n[PROGRAM] Camera released")
print("[PROGRAM] Program finished")
