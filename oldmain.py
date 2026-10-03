import os
import sys
import cv2
import json
import time
import uuid
import base64
import queue
import threading
import multiprocessing as mp
from datetime import datetime

import numpy as np
import requests
import torch
import torch.nn as nn
from ultralytics import YOLO

# ============================================================
# ARISE AI - REALTIME MULTI-AI
# ============================================================
print()
print("=" * 70)
print("ARISE AI - REALTIME MULTI-AI")
print("=" * 70)
print()

# ============================================================
# PATH CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "configAI.json")
CAMERA_CONFIG_PATH = os.path.join(BASE_DIR, "camera.json")


# ============================================================
# LOAD JSON
# ============================================================
def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


config = load_json(CONFIG_PATH)
camera_config = load_json(CAMERA_CONFIG_PATH)


# ============================================================
# CAMERA SELECTION
# ============================================================
def select_camera(camera_data):
    print("=" * 70)
    print("ARISE CAMERA SELECTION")
    print("=" * 70)

    for camera_id, camera in camera_data.items():
        print(
            f"[{camera_id}] "
            f"{camera.get('name', 'Unknown')} | "
            f"{camera.get('location', '-')} | "
            f"{camera.get('building', '-')} | "
            f"{camera.get('floor', '-')}"
        )

    print("=" * 70)

    while True:
        selected = input("เลือกหมายเลขกล้อง: ").strip()
        if selected in camera_data:
            break
        print("[ERROR] ไม่พบกล้องหมายเลขนี้")

    camera = camera_data[selected]
    camera_info = {
        "id": str(selected),
        "name": camera.get("name", f"Camera {selected}"),
        "latitude": camera.get("latitude", 0.0),
        "longitude": camera.get("longitude", 0.0),
        "location": camera.get("location", ""),
        "building": camera.get("building", ""),
        "floor": camera.get("floor", ""),
        "description": camera.get("description", ""),
    }

    print()
    print("-" * 70)
    print("Selected Camera")
    print("-" * 70)
    print(f"ID          : {camera_info['id']}")
    print(f"Name        : {camera_info['name']}")
    print(f"Location    : {camera_info['location']}")
    print(f"Building    : {camera_info['building']}")
    print(f"Floor       : {camera_info['floor']}")
    print(f"Description : {camera_info['description']}")
    print(f"Latitude    : {camera_info['latitude']}")
    print(f"Longitude   : {camera_info['longitude']}")
    print("-" * 70)
    print()

    return camera_info


# ============================================================
# UTILITY
# ============================================================
def now_iso():
    return datetime.now().isoformat()


def frame_to_base64(frame):
    if frame is None:
        return None
    try:
        success, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if not success:
            return None
        return base64.b64encode(encoded.tobytes()).decode("utf-8")
    except Exception as e:
        print(f"[Base64] Error: {e}")
        return None


def create_incident_id():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_part = uuid.uuid4().hex[:6]
    return f"{timestamp}_{random_part}"


# ============================================================
# ALERT SENDER
# ============================================================
class AlertSender:

    def __init__(self, config, camera_info):
        alert_config = config.get("web_alert", {})
        self.enabled = alert_config.get("enabled", True)
        self.url = alert_config.get("url", "http://localhost:5500/api/alerts")
        self.timeout = alert_config.get("timeout_sec", 2.0)
        self.camera = camera_info
        self.last_sent = {}
        self.cooldown = 5.0

    def can_send(self, event_type, class_name):
        key = f"{self.camera['id']}:{event_type}:{class_name}"
        current_time = time.time()
        last_time = self.last_sent.get(key, 0)
        if current_time - last_time < self.cooldown:
            return False
        self.last_sent[key] = current_time
        return True

    def send_incident(self, event_type, class_name, confidence, frame, extra_data=None):
        if not self.enabled:
            return
        if not self.can_send(event_type, class_name):
            return

        incident_id = create_incident_id()
        image_base64 = frame_to_base64(frame)

        payload = {
            "incident_id": incident_id,
            "timestamp": now_iso(),
            "status": "active",
            "event_type": event_type,
            "class_name": class_name,
            "confidence": float(confidence),
            "camera": self.camera,
            "image": image_base64,
            "data": extra_data or {},
        }

        def send():
            try:
                response = requests.post(self.url, json=payload, timeout=self.timeout)
                print()
                print("=" * 70)
                print("🚨 INCIDENT SENT")
                print("=" * 70)
                print(f"Incident ID : {incident_id}")
                print(f"Type        : {event_type}")
                print(f"Class       : {class_name}")
                print(f"Confidence  : {confidence:.2f}")
                print(f"Camera      : {self.camera['id']}")
                print(f"HTTP        : {response.status_code}")
                print("=" * 70)
            except Exception as e:
                print(f"[AlertSender] Failed: {e}")

        threading.Thread(target=send, daemon=True).start()


# ============================================================
# OBJECT DETECTOR
# ============================================================
class ObjectDetector:

    def __init__(self, config):
        model_config = config["models"]["object_detector"]
        self.path = model_config["path"]
        self.conf = float(model_config.get("conf_threshold", 0.45))
        self.classes = model_config.get("classes", {})

        print("[ObjectDetector]")
        print(f"Model: {self.path}")

        self.model = YOLO(self.path)

    def infer(self, frame):
        detections = []
        display = frame.copy()

        try:
            results = self.model.predict(frame, conf=self.conf, verbose=False)
            if not results:
                return detections, display

            result = results[0]
            display = result.plot(img=frame.copy())

            if result.boxes is None:
                return detections, display

            boxes = result.boxes

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                class_cfg = self.classes.get(str(class_id), {})
                class_name = class_cfg.get(
                    "name", self.model.names.get(class_id, str(class_id))
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                })

        except Exception as e:
            print(f"[ObjectDetector] Error: {e}")

        return detections, display


# ============================================================
# POSE ESTIMATOR + BYTETRACK
# ============================================================
class PoseEstimator:

    def __init__(self, config):
        model_config = config["models"]["pose_estimator"]
        self.path = model_config["path"]
        self.conf = float(model_config.get("conf_threshold", 0.4))
        self.num_keypoints = int(model_config.get("num_keypoints", 17))

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
                verbose=False,
            )

            if not results:
                return persons, display

            result = results[0]
            display = result.plot(img=frame.copy())

            if result.keypoints is None or result.boxes is None:
                return persons, display

            keypoints_xy = result.keypoints.xy.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()

            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [-1 for _ in range(len(keypoints_xy))]

            for i in range(len(keypoints_xy)):
                persons.append({
                    "track_id": int(track_ids[i]),
                    "keypoints": keypoints_xy[i].tolist(),
                    "bbox": boxes[i].tolist(),
                })

        except Exception as e:
            print(f"[PoseEstimator] Error: {e}")

        return persons, display


# ============================================================
# GRU MODEL
# ============================================================
class PoseGRUModel(nn.Module):

    def __init__(
        self,
        input_size=34,
        hidden_size=64,
        num_layers=2,
        num_classes=6,
        dropout=0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, _ = self.gru(x)
        last = output[:, -1, :]
        return self.fc(last)


# ============================================================
# ACTION CLASSIFIER
# ============================================================
class ActionClassifier:

    def __init__(self, config):
        model_config = config["models"]["action_classifier"]
        self.path = model_config["path"]
        self.sequence_length = int(model_config.get("sequence_length", 30))
        self.classes = model_config.get("classes", {})
        self.device = torch.device("cpu")

        print()
        print("[ActionClassifier]")
        print(f"Model: {self.path}")

        # Load checkpoint
        checkpoint = torch.load(self.path, map_location=self.device)

        # Determine number of classes
        num_classes = len(self.classes)
        if num_classes <= 0:
            num_classes = 6

        # Create model
        self.model = PoseGRUModel(
            input_size=34,
            hidden_size=64,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.2,
        )

        # Resolve state dict
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        try:
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"[ActionClassifier] State dict warning: {e}")
            # Try strict=False
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

        # Sequence buffer per track
        self.buffers = {}

    def normalize_keypoints(self, keypoints, width, height):
        keypoints = np.asarray(keypoints, dtype=np.float32)

        # Expected: (17, 2)
        if keypoints.shape[0] != 17:
            fixed = np.zeros((17, 2), dtype=np.float32)
            count = min(17, keypoints.shape[0])
            fixed[:count] = keypoints[:count]
            keypoints = fixed

        keypoints[:, 0] /= max(width, 1)   # Normalize X
        keypoints[:, 1] /= max(height, 1)  # Normalize Y
        keypoints = np.clip(keypoints, 0.0, 1.0)

        return keypoints.reshape(-1)

    def update(self, pose_results, frame_shape=None):
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

            if not isinstance(person, dict):
                continue

            track_id = person.get("track_id", -1)
            if track_id is None:
                continue

            track_id = int(track_id)
            if track_id < 0:
                continue

            keypoints = person.get("keypoints")
            if keypoints is None:
                continue

            active_tracks.add(track_id)

            feature = self.normalize_keypoints(keypoints, width, height)

            if track_id not in self.buffers:
                self.buffers[track_id] = []

            self.buffers[track_id].append(feature)

            # Keep only latest sequence
            if len(self.buffers[track_id]) > self.sequence_length:
                self.buffers[track_id] = self.buffers[track_id][-self.sequence_length:]

            # Wait until buffer is full (30 frames)
            if len(self.buffers[track_id]) < self.sequence_length:
                continue

            sequence = np.asarray(self.buffers[track_id], dtype=np.float32)
            tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)

            # Predict
            try:
                with torch.no_grad():
                    logits = self.model(tensor)
                    probabilities = torch.softmax(logits, dim=1)
                    confidence, class_id = torch.max(probabilities, dim=1)
                    class_id = int(class_id.item())
                    confidence = float(confidence.item())

                class_config = self.classes.get(str(class_id), {})
                class_name = class_config.get("name", str(class_id))

                results.append({
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                })

            except Exception as e:
                print(f"[ActionClassifier] Prediction error: {e}")

        # Remove buffers for tracks no longer active
        old_tracks = list(self.buffers.keys())
        for track_id in old_tracks:
            if track_id not in active_tracks:
                # Keep buffer for a short while; could be changed later
                pass

        return results


# ============================================================
# YAMNET WORKER
# ============================================================
def yamnet_worker(input_queue, output_queue, sample_rate, chunk_duration):
    try:
        import sounddevice as sd
        import tensorflow as tf
        import tensorflow_hub as hub

        print("[Audio] Loading YAMNet...")

        yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

        # YAMNet class names
        class_map_path = tf.keras.utils.get_file(
            "yamnet_class_map.csv",
            "https://raw.githubusercontent.com/"
            "tensorflow/models/master/"
            "research/audioset/yamnet/"
            "yamnet_class_map.csv",
        )

        class_names = []

        with open(class_map_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",", 2)
                if len(parts) >= 3:
                    class_names.append(parts[2].strip())

        print("[Audio] YAMNet ready")

        # Audio stream
        block_size = int(sample_rate * chunk_duration)

        while True:
            try:
                audio = sd.rec(
                    block_size,
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()

                audio = audio.reshape(-1)

                if len(audio) == 0:
                    output_queue.put({"class_name": "No Audio", "confidence": 0.0})
                    continue

                scores, embeddings, spectrogram = yamnet(
                    tf.convert_to_tensor(audio, dtype=tf.float32)
                )

                scores_np = scores.numpy()
                mean_scores = np.mean(scores_np, axis=0)
                class_id = int(np.argmax(mean_scores))
                confidence = float(mean_scores[class_id])

                if class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = str(class_id)

                output_queue.put({"class_name": class_name, "confidence": confidence})

            except Exception as e:
                output_queue.put({"class_name": "No Audio", "confidence": 0.0})
                print(f"[Audio] Error: {e}")

    except Exception as e:
        print(f"[Audio] Worker startup error: {e}")
        output_queue.put({"class_name": "No Audio", "confidence": 0.0})


# ============================================================
# AUDIO CONTROLLER
# ============================================================
class AudioController:

    def __init__(self, config):
        audio_config = config.get("audio", {})
        self.sample_rate = int(audio_config.get("sample_rate", 16000))
        self.chunk_duration = float(audio_config.get("chunk_duration_sec", 0.975))
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.process = None
        self.latest = {"class_name": "No Audio", "confidence": 0.0}

    def start(self):
        print("[Audio] Starting process...")
        self.process = mp.Process(
            target=yamnet_worker,
            args=(
                self.input_queue,
                self.output_queue,
                self.sample_rate,
                self.chunk_duration,
            ),
            daemon=True,
        )
        self.process.start()

    def get_latest(self):
        try:
            while True:
                self.latest = self.output_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception:
            pass
        return self.latest

    def stop(self):
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)


# ============================================================
# AI OUTPUT AGGREGATOR
# ============================================================
class AIOutputAggregator:

    def __init__(self, camera_info):
        self.camera = camera_info

    def build(self, objects, actions, audio):
        return {
            "timestamp": now_iso(),
            "camera": self.camera,
            "object_detection": objects,
            "action_gru": actions,
            "audio_yamnet": audio,
        }


# ============================================================
# OBJECT ALERT
# ============================================================
def check_object_alert(detections, config):
    alerts = []
    object_config = config["models"]["object_detector"]["classes"]

    if not detections:
        return alerts

    for detection in detections:
        if not isinstance(detection, dict):
            continue

        class_id = detection.get("class_id")
        class_name = detection.get("class_name", "unknown")
        confidence = float(detection.get("confidence", 0.0))

        class_cfg = object_config.get(str(class_id), {})
        alert_enabled = class_cfg.get("alert", False)
        threshold = float(class_cfg.get("threshold", 0.0))

        if alert_enabled and confidence >= threshold:
            alerts.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": detection.get("bbox"),
            })

    return alerts


# ============================================================
# ACTION ALERT
# ============================================================
def check_action_alert(action_results, config):
    """
    ตรวจสอบ Action GRU

    action_results:
    [
        {
            "track_id": 1,
            "class_id": 0,
            "class_name": "boxing",
            "confidence": 0.85
        }
    ]
    """
    alerts = []

    # ไม่มี action
    if not action_results:
        return alerts

    action_config = config["models"]["action_classifier"]["classes"]

    # action ต้องเป็น dict
    for action in action_results:
        if not isinstance(action, dict):
            print(
                "[WARNING] Invalid action "
                f"result: {action} "
                f"({type(action).__name__})"
            )
            continue

        class_id = action.get("class_id")
        class_name = action.get("class_name", "unknown")
        confidence = float(action.get("confidence", 0.0))
        track_id = action.get("track_id")

        # Config keys are strings
        class_cfg = action_config.get(str(class_id))

        if class_cfg is None:
            print(f"[WARNING] Action class {class_id} not found in config")
            continue

        alert_enabled = class_cfg.get("alert", False)
        threshold = float(class_cfg.get("threshold", 0.0))

        # Alert condition
        if alert_enabled and confidence >= threshold:
            alerts.append({
                "track_id": track_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
            })

    return alerts


# ============================================================
# AUDIO ALERT
# ============================================================
def check_audio_alert(audio, config):
    alerts = []

    if not isinstance(audio, dict):
        return alerts

    class_name = audio.get("class_name", "No Audio")
    confidence = float(audio.get("confidence", 0.0))

    audio_config = config["models"]["audio_classifier"]["classes"]
    class_cfg = audio_config.get(class_name)

    if class_cfg is None:
        return alerts

    alert_enabled = class_cfg.get("alert", False)
    threshold = float(class_cfg.get("threshold", 0.0))

    if alert_enabled and confidence >= threshold:
        alerts.append({"class_name": class_name, "confidence": confidence})

    return alerts


# ============================================================
# EVENT CORRELATION ENGINE
# ============================================================
class EventCorrelationEngine:
    """
    รวมเหตุการณ์จาก

        Object Detection
        Action GRU
        YAMNet Audio

    แล้วตรวจสอบว่าเหตุการณ์หลายประเภท
    เกิดขึ้นภายในช่วงเวลาเดียวกันหรือไม่
    """

    def __init__(self, config):
        correlation_config = config.get("event_correlation", {})
        self.enabled = correlation_config.get("enabled", True)
        self.window_sec = float(correlation_config.get("window_sec", 5.0))
        self.cooldown_sec = float(correlation_config.get("cooldown_sec", 10.0))
        self.rules = correlation_config.get("rules", {})

        # เก็บ event ย้อนหลัง
        self.history = []

        # เวลาที่ส่ง incident ล่าสุด
        self.last_incident_time = 0

    def add_event(self, event_type, class_name, confidence, track_id=None, extra_data=None):
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "class_name": class_name,
            "confidence": float(confidence),
            "track_id": track_id,
            "data": extra_data or {},
        }
        self.history.append(event)

    def cleanup(self):
        cutoff = time.time() - self.window_sec
        self.history = [e for e in self.history if e["timestamp"] >= cutoff]

    def add_events(self, object_alerts, action_alerts, audio_alerts):
        # Object
        for alert in object_alerts:
            self.add_event(
                event_type="object",
                class_name=alert.get("class_name", "unknown"),
                confidence=alert.get("confidence", 0.0),
                track_id=None,
                extra_data={
                    "class_id": alert.get("class_id"),
                    "bbox": alert.get("bbox"),
                },
            )

        # Action
        for alert in action_alerts:
            self.add_event(
                event_type="action",
                class_name=alert.get("class_name", "unknown"),
                confidence=alert.get("confidence", 0.0),
                track_id=alert.get("track_id"),
                extra_data={"class_id": alert.get("class_id")},
            )

        # Audio
        for alert in audio_alerts:
            self.add_event(
                event_type="audio",
                class_name=alert.get("class_name", "unknown"),
                confidence=alert.get("confidence", 0.0),
            )

        self.cleanup()

    def calculate_score(self):
        if not self.history:
            return 0.0

        # หา confidence สูงสุดของแต่ละประเภท
        # ไม่เอา frame ซ้ำ ๆ มาบวกจนคะแนนพุ่ง
        best_by_type = {}

        for event in self.history:
            event_type = event["event_type"]
            confidence = event["confidence"]

            if event_type not in best_by_type:
                best_by_type[event_type] = event
            elif confidence > best_by_type[event_type]["confidence"]:
                best_by_type[event_type] = event

        # Score จากประเภทเหตุการณ์
        scores = [event["confidence"] for event in best_by_type.values()]

        if not scores:
            return 0.0

        return float(sum(scores) / len(scores))

    def get_event_types(self):
        return set(event["event_type"] for event in self.history)

    def get_people(self):
        people = set()
        for event in self.history:
            if event["event_type"] == "action":
                track_id = event.get("track_id")
                if track_id is not None:
                    people.add(track_id)
        return people

    def evaluate(self):
        if not self.enabled:
            return None

        self.cleanup()

        if not self.history:
            return None

        current_time = time.time()

        # Cooldown
        if current_time - self.last_incident_time < self.cooldown_sec:
            return None

        event_types = self.get_event_types()
        score = self.calculate_score()

        # ====================================================
        # RULE 1: SEVERE DISTURBANCE (Object + Action + Audio)
        # ====================================================
        severe_config = self.rules.get("severe_disturbance", {})

        if severe_config.get("enabled", True):
            min_event_types = int(severe_config.get("min_event_types", 3))
            min_score = float(severe_config.get("min_score", 0.50))

            if len(event_types) >= min_event_types and score >= min_score:
                self.last_incident_time = current_time
                return {
                    "type": "severe_disturbance",
                    "score": score,
                    "event_types": list(event_types),
                    "events": list(self.history),
                    "reason": "พบ Object + Action + Audio ภายในช่วงเวลาเดียวกัน",
                }

        # ====================================================
        # RULE 2: DISTURBANCE (อย่างน้อย 2 ประเภท)
        # ====================================================
        disturbance_config = self.rules.get("disturbance", {})

        if disturbance_config.get("enabled", True):
            min_event_types = int(disturbance_config.get("min_event_types", 2))
            min_score = float(disturbance_config.get("min_score", 0.55))

            if len(event_types) >= min_event_types and score >= min_score:
                self.last_incident_time = current_time
                return {
                    "type": "disturbance",
                    "score": score,
                    "event_types": list(event_types),
                    "events": list(self.history),
                    "reason": "พบเหตุการณ์ผิดปกติ อย่างน้อย 2 ประเภท ภายในช่วงเวลาเดียวกัน",
                }

        # ====================================================
        # RULE 3: MULTIPLE PEOPLE (หลายคนทำ action ที่เป็น alert)
        # ====================================================
        people_config = self.rules.get("multiple_people", {})

        if people_config.get("enabled", True):
            min_people = int(people_config.get("min_people", 2))
            min_score = float(people_config.get("min_score", 0.60))
            people = self.get_people()

            if len(people) >= min_people and score >= min_score:
                self.last_incident_time = current_time
                return {
                    "type": "multiple_people",
                    "score": score,
                    "event_types": list(event_types),
                    "people": list(people),
                    "events": list(self.history),
                    "reason": "พบหลายบุคคลมี Action ผิดปกติภายในช่วงเวลาเดียวกัน",
                }

        return None


# ============================================================
# DRAW INFORMATION
# ============================================================
def draw_camera_info(frame, camera_info):
    y = 30
    lines = [
        f"ARISE AI | Camera {camera_info['id']}",
        f"{camera_info['location']} | "
        f"{camera_info['building']} | "
        f"{camera_info['floor']}",
    ]

    for line in lines:
        cv2.putText(
            frame, line, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )
        y += 30


def draw_actions(frame, actions):
    y = 100

    for action in actions:
        if not isinstance(action, dict):
            continue

        track_id = action.get("track_id", -1)
        class_name = action.get("class_name", "unknown")
        confidence = float(action.get("confidence", 0))

        text = f"ID {track_id}: {class_name} {confidence:.2f}"

        cv2.putText(
            frame, text, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA,
        )
        y += 28


def draw_audio(frame, audio):
    if not isinstance(audio, dict):
        return

    class_name = audio.get("class_name", "No Audio")
    confidence = float(audio.get("confidence", 0))
    text = f"Audio: {class_name} {confidence:.2f}"
    height = frame.shape[0]

    cv2.putText(
        frame, text, (20, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )


# ============================================================
# POST AI DATA
# ============================================================
def send_ai_data(payload, url="http://localhost:5500/api/ai"):
    try:
        requests.post(url, json=payload, timeout=1.5)
    except Exception:
        # Don't spam terminal every frame
        pass


# ============================================================
# MAIN
# ============================================================
def main():
    camera_info = None
    audio_controller = None
    cap = None

    try:
        # SELECT CAMERA
        camera_info = select_camera(camera_config)

        # CREATE AI MODELS
        object_detector = ObjectDetector(config)
        pose_estimator = PoseEstimator(config)
        action_classifier = ActionClassifier(config)

        # AUDIO
        audio_controller = AudioController(config)
        audio_controller.start()

        # ALERT
        alert_sender = AlertSender(config, camera_info)
        event_engine = EventCorrelationEngine(config)

        # AGGREGATOR
        aggregator = AIOutputAggregator(camera_info)

        # CAMERA CONFIG
        camera_settings = config.get("camera", {})
        device_index = int(camera_settings.get("device_index", 0))
        width = int(camera_settings.get("width", 1280))
        height = int(camera_settings.get("height", 720))
        fps = int(camera_settings.get("fps", 30))

        # OPEN CAMERA
        print()
        print(f"[Camera] Device: {device_index}")
        print(f"[Camera] Resolution: {width}x{height}")
        print(f"[Camera] FPS: {fps}")

        cap = cv2.VideoCapture(device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        if not cap.isOpened():
            raise RuntimeError(f"ไม่สามารถเปิด Camera {device_index}")

        # START
        print()
        print("=" * 70)
        print("✓ ARISE AI STARTED")
        print("=" * 70)
        print(f"Camera ID : {camera_info['id']}")
        print(f"Location  : {camera_info['location']}")
        print(f"Building  : {camera_info['building']}")
        print(f"Floor     : {camera_info['floor']}")
        print()
        print("กด Q เพื่อออก")
        print("=" * 70)

        # LOOP
        last_json_time = 0
        ai_api_url = "http://localhost:5500/api/ai"

        while True:
            ret, frame = cap.read()

            if not ret:
                print("[Camera] ไม่สามารถอ่าน frame")
                break

            # OBJECT DETECTION
            object_results, object_frame = object_detector.infer(frame)

            # OBJECT ALERT
            object_alerts = check_object_alert(object_results, config)

            # POSE
            # สำคัญ: ใช้ object_frame เพื่อให้ object box ไม่หาย
            pose_results, pose_frame = pose_estimator.infer(object_frame)

            # GRU
            action_results = []
            try:
                action_results = action_classifier.update(pose_results, frame.shape)
            except Exception as e:
                print(f"[ActionClassifier] Update error: {e}")
                action_results = []

            # ACTION ALERT
            action_alerts = check_action_alert(action_results, config)

            # AUDIO
            audio_result = audio_controller.get_latest()

            # AUDIO ALERT
            audio_alerts = check_audio_alert(audio_result, config)

            # DISPLAY FRAME
            display = pose_frame.copy()
            draw_camera_info(display, camera_info)
            draw_actions(display, action_results)
            draw_audio(display, audio_result)

            # EVENT CORRELATION
            event_engine.add_events(
                object_alerts=object_alerts,
                action_alerts=action_alerts,
                audio_alerts=audio_alerts,
            )

            correlation_result = event_engine.evaluate()

            # SEND CORRELATED INCIDENT
            if correlation_result is not None:
                event_type = correlation_result["type"]
                score = float(correlation_result["score"])
                event_types = correlation_result.get("event_types", [])

                print()
                print("=" * 70)
                print("🚨 CORRELATED EVENT DETECTED")
                print("=" * 70)
                print(f"Type       : {event_type}")
                print(f"Score      : {score:.2f}")
                print(f"Event Types: {event_types}")
                print(f"Reason     : {correlation_result.get('reason')}")
                print("=" * 70)

                alert_sender.send_incident(
                    event_type=event_type,
                    class_name=event_type,
                    confidence=score,
                    frame=display,
                    extra_data={
                        "risk_score": score,
                        "event_types": event_types,
                        "people": correlation_result.get("people", []),
                        "reason": correlation_result.get("reason", ""),
                        "evidence": correlation_result.get("events", []),
                    },
                )

            # BUILD JSON
            payload = aggregator.build(
                objects=object_results,
                actions=action_results,
                audio=audio_result,
            )

            # SEND JSON EVERY 1 SECOND
            current_time = time.time()

            if current_time - last_json_time >= 1.0:
                print()
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                send_ai_data(payload, ai_api_url)
                last_json_time = current_time

            # DISPLAY
            cv2.imshow("ARISE AI", display)

            # KEYBOARD
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print()
        print("[ARISE] Interrupted")

    except Exception as e:
        print()
        print("=" * 70)
        print("[ARISE ERROR]")
        print(str(e))
        print("=" * 70)

        import traceback
        traceback.print_exc()

    finally:
        print()
        print("Cleaning up...")

        # CAMERA
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        # AUDIO
        if audio_controller is not None:
            try:
                audio_controller.stop()
            except Exception:
                pass

        # OPENCV
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("ARISE stopped.")


# ============================================================
# MULTIPROCESSING ENTRY POINT
# ============================================================
if __name__ == "__main__":
    mp.freeze_support()

    # Linux: fork จะเร็วกว่า แต่ YAMNet/TensorFlow
    # สามารถมีปัญหากับการ fork หลัง import
    # ใช้ spawn เพื่อให้ audio process แยก environment ชัดเจน
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()