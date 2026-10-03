import os
from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, config):

        model_config = config["models"]["object_detector"]

        self.path = model_config["path"]
        self.conf = float(
            model_config.get("conf_threshold", 0.45)
        )
        self.classes = model_config.get("classes", {})

        print("[ObjectDetector]")
        print(f"Model: {self.path}")

        self.model = YOLO(self.path)

    def infer(self, frame):

        detections = []
        display = frame.copy()

        try:

            results = self.model.predict(
                frame,
                conf=self.conf,
                verbose=False
            )

            if not results:
                return detections, display

            result = results[0]

            display = result.plot(
                img=frame.copy()
            )

            if result.boxes is None:
                return detections, display

            boxes = result.boxes

            for i in range(len(boxes)):

                class_id = int(
                    boxes.cls[i].item()
                )

                confidence = float(
                    boxes.conf[i].item()
                )

                xyxy = boxes.xyxy[i].cpu().numpy()

                x1, y1, x2, y2 = map(
                    int,
                    xyxy
                )

                class_cfg = self.classes.get(
                    str(class_id),
                    {}
                )

                class_name = class_cfg.get(
                    "name",
                    self.model.names.get(
                        class_id,
                        str(class_id)
                    )
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                })

        except Exception as e:

            print(
                f"[ObjectDetector] Error: {e}"
            )

        return detections, display