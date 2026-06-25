import cv2
import os
import time
import logging
import threading
from dotenv import load_dotenv
from ultralytics import YOLO

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("aquaguard.log")
    ]
)
log = logging.getLogger("AquaGuard")

load_dotenv()

CAMERA_IP = os.getenv("CAMERA_IP")
CAMERA_USER = os.getenv("CAMERA_USER", "admin")
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD")
TRIGGER_MODE = os.getenv("TRIGGER_MODE", "relay")
GPIO_PIN = int(os.getenv("GPIO_PIN", "17"))
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.6"))
COOLDOWN = float(os.getenv("COOLDOWN", "3.0"))
BURST_DURATION = float(os.getenv("BURST_DURATION", "0.8"))
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8s.pt")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
_classes_raw = os.getenv("TARGET_CLASSES", "")
TARGET_CLASSES = [c.strip().lower()
                  for c in _classes_raw.split(",") if c.strip()]

if not CAMERA_IP or not CAMERA_PASSWORD:
    log.error("CAMERA_IP oder CAMERA_PASSWORD fehlt in .env")
    exit(1)

RTSP_URL = f"rtsp://{CAMERA_USER}:{CAMERA_PASSWORD}@{CAMERA_IP}/stream1"

# --- GPIO ---
GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    if TRIGGER_MODE == "servo":
        GPIO.setup(GPIO_PIN, GPIO.OUT)
        servo = GPIO.PWM(GPIO_PIN, 50)
        servo.start(0)
    else:
        GPIO.setup(GPIO_PIN, GPIO.OUT)
        GPIO.output(GPIO_PIN, GPIO.LOW)
    GPIO_AVAILABLE = True
    log.info(f"GPIO initialisiert · Pin {GPIO_PIN} · Modus: {TRIGGER_MODE}")
except ImportError:
    log.warning("RPi.GPIO nicht verfügbar — Simulationsmodus")


# --- Stream-Thread: hält immer den neuesten Frame bereit ---
class CameraStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


# --- Trigger ---
def fire(label, confidence):
    log.info(f"🎯 TREFFER: {label} ({confidence:.0%}) → WASSER!")
    if DRY_RUN:
        print(f"SHOOT: {label} ({confidence:.0%})", flush=True)
        return
    if not GPIO_AVAILABLE:
        return
    if TRIGGER_MODE == "servo":
        def set_angle(a):
            servo.ChangeDutyCycle(2 + a/18)
            time.sleep(0.3)
            servo.ChangeDutyCycle(0)
        set_angle(45)
        time.sleep(BURST_DURATION)
        set_angle(0)
    else:
        GPIO.output(GPIO_PIN, GPIO.HIGH)
        time.sleep(BURST_DURATION)
        GPIO.output(GPIO_PIN, GPIO.LOW)


# --- Main ---
def main():
    log.info("AquaGuard startet...")
    log.info(
        f"Modell: {MODEL_PATH} | Konfidenz: {CONFIDENCE} | Cooldown: {COOLDOWN}s")
    if DRY_RUN:
        log.info("🧪 DRY_RUN aktiv")
    if TARGET_CLASSES:
        log.info(f"Ziel-Klassen: {', '.join(TARGET_CLASSES)}")

    model = YOLO(MODEL_PATH)
    log.info(f"YOLO geladen · {len(model.names)} Klassen")

    stream = CameraStream(RTSP_URL)
    log.info("Warte auf ersten Frame...")

    # Warten bis Stream bereit
    for _ in range(30):
        if stream.get_frame() is not None:
            break
        time.sleep(0.5)
    else:
        log.error("Kein Frame empfangen — IP/Passwort prüfen")
        stream.stop()
        exit(1)

    log.info("Stream aktiv · Erkennung läuft...")
    last_trigger = {}

    try:
        while True:
            frame = stream.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            results = model(frame, verbose=False)[0]

            for box in results.boxes:
                label = model.names[int(box.cls)]
                confidence = float(box.conf)

                if confidence < CONFIDENCE:
                    continue
                if TARGET_CLASSES and label.lower() not in TARGET_CLASSES:
                    continue

                now = time.time()
                if now - last_trigger.get(label, 0) < COOLDOWN:
                    continue

                fire(label, confidence)
                last_trigger[label] = now

            time.sleep(0.1)

    except KeyboardInterrupt:
        log.info("Beendet durch Benutzer (Ctrl+C)")
    finally:
        stream.stop()
        if GPIO_AVAILABLE:
            if TRIGGER_MODE == "servo":
                servo.stop()
            GPIO.cleanup()
        log.info("Auf Wiedersehen 💧")


if __name__ == "__main__":
    main()
