from ultralytics import YOLO


class PoseEstimator:

    def __init__(self, config):

        model_config = config["models"]["pose_estimator"]

        self.path = model_config["path"]

        self.conf = float(
            model_config.get(
                "conf_threshold",
                0.4
            )
        )

        self.num_keypoints = int(
            model_config.get(
                "num_keypoints",
                17
            )
        )

        print()
        print("[PoseEstimator]")
        print(f"Model: {self.path}")

        self.model = YOLO(self.path)

    def infer(self, frame):

        persons = []
        display = frame.copy()

        try:

            results = self.model.track(
                frame,
                conf=self.conf,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            if not results:
                return persons, display

            result = results[0]

            display = result.plot(
                img=frame.copy()
            )

            if (
                result.keypoints is None
                or result.boxes is None
            ):
                return persons, display

            keypoints_xy = (
                result.keypoints.xy
                .cpu()
                .numpy()
            )

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            if result.boxes.id is not None:

                track_ids = (
                    result.boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

            else:

                track_ids = [
                    -1
                    for _ in range(
                        len(keypoints_xy)
                    )
                ]

            for i in range(
                len(keypoints_xy)
            ):

                persons.append({

                    "track_id":
                        int(track_ids[i]),

                    "keypoints":
                        keypoints_xy[i].tolist(),

                    "bbox":
                        boxes[i].tolist()
                })

        except Exception as e:

            print(
                f"[PoseEstimator] Error: {e}"
            )

        return persons, display