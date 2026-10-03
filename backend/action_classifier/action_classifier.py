import numpy as np
import torch
import torch.nn as nn


class PoseGRUModel(nn.Module):

    def __init__(
        self,
        input_size=34,
        hidden_size=64,
        num_layers=2,
        num_classes=6,
        dropout=0.2
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        output, _ = self.gru(x)

        last = output[:, -1, :]

        return self.fc(last)


class ActionClassifier:

    def __init__(self, config):

        model_config = config[
            "models"
        ]["action_classifier"]

        self.path = model_config["path"]

        self.sequence_length = int(
            model_config.get(
                "sequence_length",
                30
            )
        )

        self.classes = model_config.get(
            "classes",
            {}
        )

        self.device = torch.device("cpu")

        print()
        print("[ActionClassifier]")
        print(f"Model: {self.path}")

        checkpoint = torch.load(
            self.path,
            map_location=self.device
        )

        num_classes = len(
            self.classes
        )

        if num_classes <= 0:
            num_classes = 6

        self.model = PoseGRUModel(
            input_size=34,
            hidden_size=64,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.2
        )

        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint[
                    "model_state_dict"
                ]

            else:
                state_dict = checkpoint

        else:
            state_dict = checkpoint

        try:

            self.model.load_state_dict(
                state_dict
            )

        except Exception as e:

            print(
                f"[ActionClassifier] "
                f"State dict warning: {e}"
            )

            self.model.load_state_dict(
                state_dict,
                strict=False
            )

        self.model.to(self.device)
        self.model.eval()

        self.buffers = {}

    def normalize_keypoints(
        self,
        keypoints,
        width,
        height
    ):

        keypoints = np.asarray(
            keypoints,
            dtype=np.float32
        )

        if keypoints.shape[0] != 17:

            fixed = np.zeros(
                (17, 2),
                dtype=np.float32
            )

            count = min(
                17,
                keypoints.shape[0]
            )

            fixed[:count] = (
                keypoints[:count]
            )

            keypoints = fixed

        keypoints[:, 0] /= max(
            width,
            1
        )

        keypoints[:, 1] /= max(
            height,
            1
        )

        keypoints = np.clip(
            keypoints,
            0.0,
            1.0
        )

        return keypoints.reshape(-1)

    def update(
        self,
        pose_results,
        frame_shape=None
    ):

        results = []

        if not pose_results:
            return results

        if frame_shape is None:

            height = 720
            width = 1280

        else:

            height, width = frame_shape[:2]

        active_tracks = set()

        for person in pose_results:

            if not isinstance(
                person,
                dict
            ):
                continue

            track_id = person.get(
                "track_id",
                -1
            )

            if track_id is None:
                continue

            track_id = int(track_id)

            if track_id < 0:
                continue

            keypoints = person.get(
                "keypoints"
            )

            if keypoints is None:
                continue

            active_tracks.add(
                track_id
            )

            feature = (
                self.normalize_keypoints(
                    keypoints,
                    width,
                    height
                )
            )

            if track_id not in self.buffers:
                self.buffers[track_id] = []

            self.buffers[
                track_id
            ].append(feature)

            if len(
                self.buffers[track_id]
            ) > self.sequence_length:

                self.buffers[track_id] = (
                    self.buffers[track_id][
                        -self.sequence_length:
                    ]
                )

            if len(
                self.buffers[track_id]
            ) < self.sequence_length:

                continue

            sequence = np.asarray(
                self.buffers[track_id],
                dtype=np.float32
            )

            tensor = (
                torch.from_numpy(
                    sequence
                )
                .unsqueeze(0)
                .to(self.device)
            )

            try:

                with torch.no_grad():

                    logits = self.model(
                        tensor
                    )

                    probabilities = (
                        torch.softmax(
                            logits,
                            dim=1
                        )
                    )

                    confidence, class_id = (
                        torch.max(
                            probabilities,
                            dim=1
                        )
                    )

                    class_id = int(
                        class_id.item()
                    )

                    confidence = float(
                        confidence.item()
                    )

                class_config = (
                    self.classes.get(
                        str(class_id),
                        {}
                    )
                )

                class_name = (
                    class_config.get(
                        "name",
                        str(class_id)
                    )
                )

                results.append({

                    "track_id":
                        track_id,

                    "class_id":
                        class_id,

                    "class_name":
                        class_name,

                    "confidence":
                        confidence
                })

            except Exception as e:

                print(
                    "[ActionClassifier] "
                    f"Prediction error: {e}"
                )

        return results