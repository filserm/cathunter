import cv2
import os
from dotenv import load_dotenv

load_dotenv()

CAMERA_IP = os.getenv("CAMERA_IP")
CAMERA_USER = os.getenv("CAMERA_USER", "admin")
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD")

if not CAMERA_IP or not CAMERA_PASSWORD:
    print("❌ Fehler: CAMERA_IP oder CAMERA_PASSWORD fehlt in der .env Datei")
    exit(1)

RTSP_URL = f"rtsp://{CAMERA_USER}:{CAMERA_PASSWORD}@{CAMERA_IP}/stream1"

print(f"🔗 Verbinde mit: rtsp://{CAMERA_USER}:***@{CAMERA_IP}/stream1")

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

if not cap.isOpened():
    print("❌ Stream konnte nicht geöffnet werden.")
    print("   → IP-Adresse und Passwort in der .env prüfen")
    print("   → Kamera und Pi im selben Netzwerk?")
    print(f"   → Ping-Test: ping {CAMERA_IP}")
    exit(1)

print("✅ Verbindung erfolgreich!")

ret, frame = cap.read()

if ret:
    h, w = frame.shape[:2]
    print(f"📐 Auflösung: {w}x{h}")
    output_path = "test_frame.jpg"
    cv2.imwrite(output_path, frame)
    print(f"📸 Testbild gespeichert: {output_path}")
else:
    print("⚠️  Stream geöffnet, aber kein Bild empfangen.")
    print("   → 'stream2' statt 'stream1' in RTSP_URL probieren")

cap.release()
print("🔌 Verbindung getrennt.")
