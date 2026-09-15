from ultralytics import YOLO

import tensorflow as tf

import cv2
import numpy as np


# =====================================
# Load Models
# =====================================

#โหลด โมเดลที่เราเทรน
yolo = YOLO(
    # "runs/pose/yolo11_pose/weights/best.pt"
    "yolo11n-pose.pt"
)

#โหลด 
gru = tf.keras.models.load_model(
    "models/gru_action.keras"
)

#โหลดว่ามี class อะไรบ้าง
classes = np.load(
    "models/gru_classes.npy",
    allow_pickle=True
)


# =====================================
# Camera
# =====================================

cap = cv2.VideoCapture(0)


sequence = []


while True:

    ret, frame = cap.read()

    if not ret:
        break


    results = yolo(
        frame,
        verbose=False
    )


    if (
        len(results) > 0
        and results[0].keypoints is not None
        and len(results[0].keypoints.xy) > 0
    ):

        keypoints = (
            results[0]
            .keypoints
            .xy[0]
            .cpu()
            .numpy()
        )


        h, w = frame.shape[:2]


        keypoints[:, 0] /= w
        keypoints[:, 1] /= h


        # (17,2)
        sequence.append(
            keypoints
        )


    # =================================
    # Keep last 30 frames
    # =================================

    if len(sequence) > 30:

        sequence.pop(0)


    # =================================
    # GRU Prediction
    # =================================

    if len(sequence) == 30:

        data = np.array(
            sequence,
            dtype=np.float32
        )


        data = data.reshape(
            1,
            30,
            34
        )


        prediction = gru.predict(
            data,
            verbose=0
        )[0]


        index = np.argmax(
            prediction
        )


        confidence = prediction[index]


        label = classes[index]


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


    cv2.imshow(
        "YOLOv11 + GRU",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()