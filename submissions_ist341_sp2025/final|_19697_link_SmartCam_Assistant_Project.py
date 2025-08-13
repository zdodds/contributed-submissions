### Data training ###
#
# All labeled photos have been uploaded to the link below on my Google Drive.
# After the data training finishes, the best weights model will be downloaded in order to use it
# with the local machine environment that allows for live cam capture.


from ultralytics import YOLO


DATA_YAML = '/content/drive/MyDrive/SmartCamAssistant/data.yaml'

model = YOLO('yolov5s.pt')  # or yolov5n.pt for a tiny model
model.train(data=DATA_YAML, epochs=30, imgsz=640)



# Download the best weights
from google.colab import files
files.download('runs/detect/train/weights/best.pt')



import time
import cv2
import pyttsx3
import numpy as np
from ultralytics import YOLO

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0       # 0 for the built-in cam, 1 is for the USB-cam
WEIGHTS_PATH = "best.pt"

# ─── OPEN CAMERA ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera at index {CAMERA_INDEX}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────────
model = YOLO(WEIGHTS_PATH)

# ─── TEXT-TO-SPEECH SETUP ─────────────────────────────────────────────────────────
tts = pyttsx3.init()
tts.setProperty("rate", tts.getProperty("rate") - 50)

# State variables for delayed announcement
_pending_color = None
_pending_time  = 0
_last_announced = None
DELAY_SECONDS   = 2.0

def announce(color):
    tts.say({
        'red':    "It is a red light, you need to stop.",
        'yellow': "Slow down, it is a yellow light.",
        'green':  "It is a green light, keep driving."
    }[color])
    tts.runAndWait()


# ─── MAIN LOOP ───────────────────────────────────────────────────────────────────
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Inference
        results = model(frame)[0]

        # 2) Find detected colors
        colors = set()
        for det in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            color = model.names[int(cls)]
            colors.add(color)

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.putText(frame, color, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # 3) Determine highest-priority color (or None)
        new_color = None
        if "red"    in colors: new_color = "red"
        elif "yellow" in colors: new_color = "yellow"
        elif "green"  in colors: new_color = "green"

        # 4) Schedule a delayed announcement if it changed
        if new_color is not None and new_color != _last_announced:
            if new_color != _pending_color:
                _pending_color = new_color
                _pending_time  = time.time()

        # 5) Check if it's time to speak
        if _pending_color is not None:
            if time.time() - _pending_time >= DELAY_SECONDS:
                announce(_pending_color)
                _pending_color = None

        # 6) Show the frame
        cv2.imshow("SmartCam Assistant", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()



