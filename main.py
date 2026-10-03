import os
import json
import time
import uuid
import base64
import threading
import multiprocessing as mp
from datetime import datetime

import cv2
import requests

from backend.object_detector.object_detector import ObjectDetector
from backend.action_classifier.pose_estimator import PoseEstimator
from backend.action_classifier.action_classifier import ActionClassifier
from backend.audio_classifier.audio_classifier import AudioController

print("=" * 70)
print("ARISE AI - MAIN CONTROLLER")
print("=" * 70)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "configAI.json")
CAMERA_CONFIG_PATH = os.path.join(BASE_DIR, "camera.json")

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_json(CONFIG_PATH)
camera_config = load_json(CAMERA_CONFIG_PATH)

def now_iso():
    return datetime.now().isoformat()

def frame_to_base64(frame):
    if frame is None:
        return None
    try:
        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
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
        "description": camera.get("description", "")
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

class AlertSender:
    def __init__(self, config, camera_info):
        alert_config = config.get("web_alert", {})
        self.enabled = alert_config.get("enabled", True)
        self.url = alert_config.get("url", "http://localhost:5500/api/alerts")
        self.timeout = float(alert_config.get("timeout_sec", 2.0))
        self.camera = camera_info
        self.last_sent = {}
        self.cooldown = float(alert_config.get("cooldown_sec", 5.0))

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
            "data": extra_data or {}
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

def check_object_alert(detections, config):

    alerts = []

    object_config = config[
        "models"
    ][
        "object_detector"
    ][
        "classes"
    ]

    if not detections:
        return alerts

    for detection in detections:

        if not isinstance(detection, dict):
            continue

        class_id = detection.get(
            "class_id"
        )

        class_name = detection.get(
            "class_name",
            "unknown"
        )

        confidence = float(
            detection.get(
                "confidence",
                0.0
            )
        )

        class_cfg = object_config.get(
            str(class_id)
        )

        if class_cfg is None:
            continue

        alert_enabled = class_cfg.get(
            "alert",
            False
        )

        threshold = float(
            class_cfg.get(
                "threshold",
                0.0
            )
        )

        # threshold 0 = ignore
        if not alert_enabled:
            continue

        if threshold <= 0:
            continue

        if confidence >= threshold:

            alerts.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": detection.get(
                    "bbox"
                )
            })

    return alerts

def check_action_alert(action_results, config):

    alerts = []

    if not action_results:
        return alerts

    action_config = config[
        "models"
    ][
        "action_classifier"
    ][
        "classes"
    ]

    for action in action_results:

        if not isinstance(action, dict):
            continue

        class_id = action.get(
            "class_id"
        )

        class_name = action.get(
            "class_name",
            "unknown"
        )

        confidence = float(
            action.get(
                "confidence",
                0.0
            )
        )

        track_id = action.get(
            "track_id"
        )

        class_cfg = action_config.get(
            str(class_id)
        )

        # Model มี class ที่ JSON ไม่รู้จัก
        # ให้ ignore เงียบ ๆ
        if class_cfg is None:
            continue

        alert_enabled = class_cfg.get(
            "alert",
            False
        )

        threshold = float(
            class_cfg.get(
                "threshold",
                0.0
            )
        )

        # threshold = 0 และ alert = false
        # ถือว่า ignore
        if not alert_enabled:
            continue

        if threshold <= 0:
            continue

        if confidence >= threshold:

            alerts.append({
                "track_id": track_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence
            })

    return alerts

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
        alerts.append({
            "class_name": class_name,
            "confidence": confidence
        })

    return alerts

class EventCorrelationEngine:

    def __init__(self, config):
        self.config = config

        correlation_config = config.get(
            "event_correlation",
            {}
        )

        self.enabled = correlation_config.get(
            "enabled",
            True
        )

        self.default_window_sec = float(
            correlation_config.get(
                "window_sec",
                5.0
            )
        )

        self.default_cooldown_sec = float(
            correlation_config.get(
                "cooldown_sec",
                10.0
            )
        )

        self.rules = correlation_config.get(
            "rules",
            {}
        )

        self.history = []

        # cooldown แยกตาม incident
        self.last_incident_time = {}

    # ---------------------------------------------------------
    # ADD EVENT
    # ---------------------------------------------------------

    def add_event(
        self,
        event_type,
        class_name,
        confidence,
        track_id=None,
        extra_data=None
    ):

        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "class_name": class_name,
            "confidence": float(confidence),
            "track_id": track_id,
            "data": extra_data or {}
        }

        self.history.append(event)

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    def cleanup(self):

        if not self.history:
            return

        # หา window ที่ยาวที่สุดจาก rules
        max_window = self.default_window_sec

        for rule in self.rules.values():

            if not isinstance(rule, dict):
                continue

            window_sec = float(
                rule.get(
                    "window_sec",
                    self.default_window_sec
                )
            )

            max_window = max(
                max_window,
                window_sec
            )

        cutoff = time.time() - max_window

        self.history = [
            event
            for event in self.history
            if event["timestamp"] >= cutoff
        ]

    # ---------------------------------------------------------
    # ADD EVENTS
    # ---------------------------------------------------------

    def add_events(
        self,
        object_alerts,
        action_alerts,
        audio_alerts
    ):

        # -------------------------
        # Object
        # -------------------------

        for alert in object_alerts:

            self.add_event(
                event_type="object",
                class_name=alert.get(
                    "class_name",
                    "unknown"
                ),
                confidence=alert.get(
                    "confidence",
                    0.0
                ),
                extra_data={
                    "class_id": alert.get(
                        "class_id"
                    ),
                    "bbox": alert.get(
                        "bbox"
                    )
                }
            )

        # -------------------------
        # Action
        # -------------------------

        for alert in action_alerts:

            self.add_event(
                event_type="action",
                class_name=alert.get(
                    "class_name",
                    "unknown"
                ),
                confidence=alert.get(
                    "confidence",
                    0.0
                ),
                track_id=alert.get(
                    "track_id"
                ),
                extra_data={
                    "class_id": alert.get(
                        "class_id"
                    )
                }
            )

        # -------------------------
        # Audio
        # -------------------------

        for alert in audio_alerts:

            self.add_event(
                event_type="audio",
                class_name=alert.get(
                    "class_name",
                    "unknown"
                ),
                confidence=alert.get(
                    "confidence",
                    0.0
                )
            )

        self.cleanup()

    # ---------------------------------------------------------
    # GET EVENTS FOR RULE
    # ---------------------------------------------------------

    def get_rule_events(self, rule):

        window_sec = float(
            rule.get(
                "window_sec",
                self.default_window_sec
            )
        )

        cutoff = time.time() - window_sec

        return [
            event
            for event in self.history
            if event["timestamp"] >= cutoff
        ]

    # ---------------------------------------------------------
    # MATCH EVENTS
    # ---------------------------------------------------------

    def get_matching_events(
        self,
        events,
        event_type,
        classes
    ):

        if not classes:
            return []

        return [
            event
            for event in events
            if (
                event["event_type"] == event_type
                and event["class_name"] in classes
            )
        ]

    # ---------------------------------------------------------
    # SIGNAL SCORE
    #
    # confidence × threshold × count
    # ---------------------------------------------------------

    def calculate_signal_score(
        self,
        events,
        threshold,
        max_count
    ):

        if not events:
            return 0.0

        threshold = float(threshold)
        max_count = int(max_count)

        # threshold 0 = ignore
        if threshold <= 0:
            return 0.0

        if max_count <= 0:
            return 0.0

        # จำกัดจำนวน event
        events = events[-max_count:]

        if not events:
            return 0.0

        # confidence เฉลี่ย
        avg_confidence = sum(
            float(event.get("confidence", 0.0))
            for event in events
        ) / len(events)

        count = min(
            len(events),
            max_count
        )

        score = (
            avg_confidence
            * threshold
            * count
        )

        # ไม่ให้เกิน 1
        return min(
            max(score, 0.0),
            1.0
        )

    # ---------------------------------------------------------
    # PEOPLE SCORE
    # ---------------------------------------------------------

    def calculate_people_score(
        self,
        events,
        signal_config,
        relevant_action_classes=None
    ):

        min_people = int(
            signal_config.get(
                "min_people",
                2
            )
        )

        if min_people <= 0:
            return 0.0

        people = set()

        for event in events:

            if event["event_type"] != "action":
                continue

            # ถ้า rule ระบุ action classes
            # ให้นับเฉพาะคนที่อยู่ใน action เหล่านั้น
            if relevant_action_classes:

                if event["class_name"] not in relevant_action_classes:
                    continue

            track_id = event.get(
                "track_id"
            )

            if track_id is not None:
                people.add(track_id)

        if len(people) < min_people:
            return 0.0

        # people signal ถือว่าเต็มเมื่อถึง min_people
        return 1.0

    # ---------------------------------------------------------
    # CALCULATE RULE SCORE
    # ---------------------------------------------------------

    def calculate_rule_score(
        self,
        rule,
        events
    ):

        signals = rule.get(
            "signals",
            {}
        )

        total_score = 0.0
        total_weight = 0.0

        # เก็บ action classes ของ rule
        action_classes = set()

        action_signal = signals.get(
            "action"
        )

        if action_signal:

            action_classes = set(
                action_signal.get(
                    "classes",
                    []
                )
            )

        # =====================================================
        # ACTION
        # =====================================================

        if action_signal:

            classes = action_signal.get(
                "classes",
                []
            )

            threshold = float(
                action_signal.get(
                    "threshold",
                    0.0
                )
            )

            weight = float(
                action_signal.get(
                    "weight",
                    0.0
                )
            )

            max_count = int(
                action_signal.get(
                    "max_count",
                    1
                )
            )

            matching_events = self.get_matching_events(
                events,
                "action",
                classes
            )

            signal_score = self.calculate_signal_score(
                matching_events,
                threshold,
                max_count
            )

            if weight > 0:

                total_score += (
                    signal_score * weight
                )

                total_weight += weight

        # =====================================================
        # OBJECT / HANDGUN
        # =====================================================

        handgun_signal = signals.get(
            "handgun"
        )

        if handgun_signal:

            classes = handgun_signal.get(
                "classes",
                []
            )

            threshold = float(
                handgun_signal.get(
                    "threshold",
                    0.0
                )
            )

            weight = float(
                handgun_signal.get(
                    "weight",
                    0.0
                )
            )

            max_count = int(
                handgun_signal.get(
                    "max_count",
                    1
                )
            )

            matching_events = self.get_matching_events(
                events,
                "object",
                classes
            )

            signal_score = self.calculate_signal_score(
                matching_events,
                threshold,
                max_count
            )

            if weight > 0:

                total_score += (
                    signal_score * weight
                )

                total_weight += weight

        # =====================================================
        # AUDIO
        # =====================================================

        audio_signal = signals.get(
            "audio"
        )

        if audio_signal:

            classes = audio_signal.get(
                "classes",
                []
            )

            threshold = float(
                audio_signal.get(
                    "threshold",
                    0.0
                )
            )

            weight = float(
                audio_signal.get(
                    "weight",
                    0.0
                )
            )

            max_count = int(
                audio_signal.get(
                    "max_count",
                    1
                )
            )

            matching_events = self.get_matching_events(
                events,
                "audio",
                classes
            )

            signal_score = self.calculate_signal_score(
                matching_events,
                threshold,
                max_count
            )

            if weight > 0:

                total_score += (
                    signal_score * weight
                )

                total_weight += weight

        # =====================================================
        # PEOPLE
        # =====================================================

        people_signal = signals.get(
            "people"
        )

        if people_signal:

            weight = float(
                people_signal.get(
                    "weight",
                    0.0
                )
            )

            signal_score = self.calculate_people_score(
                events,
                people_signal,
                action_classes
            )

            if weight > 0:

                total_score += (
                    signal_score * weight
                )

                total_weight += weight

        if total_weight <= 0:
            return 0.0

        return float(
            total_score / total_weight
        )

    # ---------------------------------------------------------
    # GET PEOPLE
    # ---------------------------------------------------------

    def get_people(
        self,
        events,
        action_classes=None
    ):

        people = set()

        for event in events:

            if event["event_type"] != "action":
                continue

            if action_classes:

                if event["class_name"] not in action_classes:
                    continue

            track_id = event.get(
                "track_id"
            )

            if track_id is not None:
                people.add(track_id)

        return people

    # ---------------------------------------------------------
    # EVALUATE ONE RULE
    # ---------------------------------------------------------

    def evaluate_rule(
        self,
        rule_name,
        rule
    ):

        if not rule.get(
            "enabled",
            True
        ):
            return None

        events = self.get_rule_events(
            rule
        )

        if not events:
            return None

        score = self.calculate_rule_score(
            rule,
            events
        )

        min_score = float(
            rule.get(
                "min_score",
                0.0
            )
        )

        if score < min_score:
            return None

        # -----------------------------------------------------
        # Cooldown ของ rule นี้
        # -----------------------------------------------------

        cooldown_sec = float(
            rule.get(
                "cooldown_sec",
                self.default_cooldown_sec
            )
        )

        current_time = time.time()

        last_time = self.last_incident_time.get(
            rule_name,
            0
        )

        if (
            current_time - last_time
            < cooldown_sec
        ):
            return None

        self.last_incident_time[
            rule_name
        ] = current_time

        # -----------------------------------------------------
        # People
        # -----------------------------------------------------

        signals = rule.get(
            "signals",
            {}
        )

        action_signal = signals.get(
            "action",
            {}
        )

        action_classes = action_signal.get(
            "classes",
            []
        )

        people = self.get_people(
            events,
            action_classes
        )

        # -----------------------------------------------------
        # Event types
        # -----------------------------------------------------

        event_types = sorted(
            set(
                event["event_type"]
                for event in events
            )
        )

        # -----------------------------------------------------
        # Reason
        # -----------------------------------------------------

        reason = (
            f"Rule '{rule_name}' matched "
            f"with score {score:.2f}"
        )

        return {
            "type": rule_name,
            "score": score,
            "event_types": event_types,
            "events": list(events),
            "people": list(people),
            "reason": reason
        }

    # ---------------------------------------------------------
    # EVALUATE
    # ---------------------------------------------------------

    def evaluate(self):

        if not self.enabled:
            return None

        self.cleanup()

        if not self.history:
            return None

        # ใช้ลำดับตาม JSON
        for rule_name, rule in self.rules.items():

            result = self.evaluate_rule(
                rule_name,
                rule
            )

            if result is not None:
                return result

        return None



def draw_camera_info(frame, camera_info):
    y = 30

    lines = [
        f"ARISE AI | Camera {camera_info['id']}",
        f"{camera_info['location']} | {camera_info['building']} | {camera_info['floor']}"
    ]

    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
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
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA
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
        frame,
        text,
        (20, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

def build_ai_payload(camera_info, object_results, action_results, audio_result):
    return {
        "timestamp": now_iso(),
        "camera": camera_info,
        "object_detection": object_results,
        "action_gru": action_results,
        "audio_yamnet": audio_result
    }

def send_ai_data(payload, url):
    try:
        response = requests.post(url, json=payload, timeout=1.5)
        return response
    except Exception:
        return None

def main():
    camera_info = None
    audio_controller = None
    cap = None

    try:
        camera_info = select_camera(camera_config)

        print("[MAIN] Loading AI modules...")

        object_detector = ObjectDetector(config)
        pose_estimator = PoseEstimator(config)
        action_classifier = ActionClassifier(config)
        audio_controller = AudioController(config)

        audio_controller.start()

        alert_sender = AlertSender(config, camera_info)
        event_engine = EventCorrelationEngine(config)

        camera_settings = config.get("camera", {})

        device_index = int(camera_settings.get("device_index", 0))
        width = int(camera_settings.get("width", 1280))
        height = int(camera_settings.get("height", 720))
        fps = int(camera_settings.get("fps", 30))

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

        ai_api_url = config.get("web_api", {}).get(
            "ai_url",
            "http://localhost:5500/api/ai"
        )

        last_json_time = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                print("[Camera] ไม่สามารถอ่าน frame")
                break

            object_results, object_frame = object_detector.infer(frame)
            object_alerts = check_object_alert(object_results, config)

            pose_results, pose_frame = pose_estimator.infer(object_frame)

            try:
                action_results = action_classifier.update(
                    pose_results,
                    frame.shape
                )
            except Exception as e:
                print(f"[ActionClassifier] Update error: {e}")
                action_results = []

            action_alerts = check_action_alert(action_results, config)

            audio_result = audio_controller.get_latest()
            audio_alerts = check_audio_alert(audio_result, config)

            display = pose_frame.copy()

            draw_camera_info(display, camera_info)
            draw_actions(display, action_results)
            draw_audio(display, audio_result)

            event_engine.add_events(
                object_alerts=object_alerts,
                action_alerts=action_alerts,
                audio_alerts=audio_alerts
            )

            correlation_result = event_engine.evaluate()

            if correlation_result is not None:
                event_type = correlation_result["type"]
                score = float(correlation_result["score"])
                event_types = correlation_result.get("event_types", [])
                evidence = correlation_result.get("events", [])
                people = correlation_result.get("people", [])
                reason = correlation_result.get("reason", "")

                print()
                print("=" * 70)
                print("🚨 CORRELATED EVENT DETECTED")
                print("=" * 70)
                print(f"Type       : {event_type}")
                print(f"Score      : {score:.2f}")
                print(f"Event Types: {event_types}")
                print(f"Evidence   : {len(evidence)}")
                print(f"Reason     : {reason}")
                print("=" * 70)

                alert_sender.send_incident(
                    event_type=event_type,
                    class_name=event_type,
                    confidence=score,
                    frame=display,
                    extra_data={
                        "risk_score": score,
                        "event_types": event_types,
                        "people": people,
                        "reason": reason,
                        "evidence": evidence
                    }
                )

            payload = build_ai_payload(
                camera_info=camera_info,
                object_results=object_results,
                action_results=action_results,
                audio_result=audio_result
            )

            current_time = time.time()

            if current_time - last_json_time >= 1.0:
                print()
                print(json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2
                ))

                send_ai_data(payload, ai_api_url)
                last_json_time = current_time

            cv2.imshow("ARISE AI", display)

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

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        if audio_controller is not None:
            try:
                audio_controller.stop()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("ARISE stopped.")

if __name__ == "__main__":
    mp.freeze_support()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()