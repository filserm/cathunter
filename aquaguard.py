import cv2
import os
import time
import logging
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

# --- .env laden ---
load_dotenv()

CAMERA_IP = os.getenv("CAMERA_IP")
CAMERA_USER = os.getenv("CAMERA_USER", "admin")
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD")

# Trigger-Modus: "relay" oder "servo"
TRIGGER_MODE = os.getenv("TRIGGER_MODE", "relay")

# GPIO Pin (BCM-Nummerierung)
GPIO_PIN = int(os.getenv("GPIO_PIN", "17"))

# Erkennungs-Einstellungen
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.6"))
COOLDOWN = float(os.getenv("COOLDOWN", "3.0"))
BURST_DURATION = float(os.getenv("BURST_DURATION", "0.8"))

# YOLO-Modell
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")

# Dry-Run: kein GPIO, nur stdout-Ausgabe
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# Ziel-Klassen Whitelist: leer = alles triggert, sonst z.B. "cat,dog,bird"
_classes_raw = os.getenv("TARGET_CLASSES", "")
TARGET_CLASSES = [c.strip().lower()
                  for c in _classes_raw.split(",") if c.strip()]

# --- Validierung ---
if not CAMERA_IP or not CAMERA_PASSWORD:
    log.error("CAMERA_IP oder CAMERA_PASSWORD fehlt in der .env Datei")
    exit(1)

RTSP_URL = f"rtsp://{CAMERA_USER}:{CAMERA_PASSWORD}@{CAMERA_IP}/stream1"

# --- GPIO Setup ---
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
    log.warning(
        "RPi.GPIO nicht verfügbar — läuft im Simulationsmodus (kein Pi?)")


# --- Trigger-Funktion ---
def fire(label: str, confidence: float):
    log.info(f"🎯 TREFFER: {label} ({confidence:.0%}) → WASSER!")

    if DRY_RUN:
        print(f"SHOOT: {label} ({confidence:.0%})", flush=True)
        return

    if not GPIO_AVAILABLE:
        log.info(
            f"[SIM] Würde GPIO Pin {GPIO_PIN} für {BURST_DURATION}s aktivieren")
        return

    if TRIGGER_MODE == "servo":
        # Servo: Abzug ziehen und zurück
        def set_angle(angle):
            duty = 2 + (angle / 18)
            servo.ChangeDutyCycle(duty)
            time.sleep(0.3)
            servo.ChangeDutyCycle(0)
        set_angle(45)
        time.sleep(BURST_DURATION)
        set_angle(0)
    else:
        # Relais: Pin HIGH für Burst-Dauer
        GPIO.output(GPIO_PIN, GPIO.HIGH)
        time.sleep(BURST_DURATION)
        GPIO.output(GPIO_PIN, GPIO.LOW)


# --- Hauptschleife ---
def main():
    log.info("AquaGuard startet...")
    log.info(
        f"Modell: {MODEL_PATH} | Konfidenz: {CONFIDENCE} | Cooldown: {COOLDOWN}s | Burst: {BURST_DURATION}s")
    if DRY_RUN:
        log.info("🧪 DRY_RUN aktiv — kein GPIO, Ausgabe nur auf stdout")

    model = YOLO(MODEL_PATH)
    log.info(f"YOLO geladen · {len(model.names)} Klassen verfügbar")

    log.info(
        f"Verbinde mit Stream: rtsp://{CAMERA_USER}:***@{CAMERA_IP}/stream1")
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)

    if not cap.isOpened():
        log.error("Stream konnte nicht geöffnet werden — IP/Passwort prüfen")
        exit(1)

    log.info("Stream aktiv · Erkennung läuft...")

    last_trigger = {}  # pro Klasse getrennt
    frame_count = 0

    try:
        while True:
            # Buffer leeren: 3 Frames verwerfen, dann aktuellen holen
            for _ in range(3):
                cap.grab()

            ret, frame = cap.retrieve()

            if not ret:
                log.warning("Kein Frame empfangen — reconnecting...")
                time.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(RTSP_URL)
                continue

            frame_count += 1

            results = model(frame, verbose=False)[0]

            for box in results.boxes:
                label = model.names[int(box.cls)]
                confidence = float(box.conf)

                if confidence < CONFIDENCE:
                    continue

                if TARGET_CLASSES and label.lower() not in TARGET_CLASSES:
                    log.debug(f"{label} ignoriert (nicht in TARGET_CLASSES)")
                    continue

                now = time.time()
                last_for_label = last_trigger.get(label, 0)
                if now - last_for_label < COOLDOWN:
                    remaining = COOLDOWN - (now - last_for_label)
                    log.debug(
                        f"{label} erkannt aber Cooldown aktiv ({remaining:.1f}s)")
                    continue

                fire(label, confidence)
                last_trigger[label] = now

    except KeyboardInterrupt:
        log.info("Beendet durch Benutzer (Ctrl+C)")
    finally:
        cap.release()
        if GPIO_AVAILABLE:
            if TRIGGER_MODE == "servo":
                servo.stop()
            GPIO.cleanup()
        log.info("GPIO cleanup · Auf Wiedersehen 💧")


if __name__ == "__main__":
    main()
