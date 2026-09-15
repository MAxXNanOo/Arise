from ultralytics import YOLO


# YOLOv11 Pose
model = YOLO("yolo11n-pose.pt")


model.train(
    data="vision/dataset/data.yaml",

    epochs=100,

    imgsz=640,

    batch=16,

    workers=4,

    name="yolo11_pose"
)