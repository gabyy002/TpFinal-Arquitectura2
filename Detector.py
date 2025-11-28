# detector.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import time

app = Flask(__name__)

# Cargar modelo YOLOv8 (liviano). Esto descargará yolov8n.pt si no está.
model = YOLO("yolov8n.pt")   # 'yolov8n' es el modelo más pequeño y rápido para CPU

# Configs de rendimiento
INPUT_WIDTH = 640   # ancho para redimensionar la imagen para la red (ajustable)
CONF_THRESH = 0.20  # umbral de confianza para aceptar detecciones
PERSISTENCE_FRAMES = 5  # cuantos frames mantener detección para suavizar

# estado
persistencia = 0

@app.route("/detect", methods=["POST"])
def detect():
    global persistencia

    # Recibir bytes y decodificar imagen
    img_bytes = request.data
    if not img_bytes:
        return jsonify({"persona": False})

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"persona": False})

    # Reducir tamaño para acelerar (mantener aspecto)
    h, w = frame.shape[:2]
    if max(w, h) > INPUT_WIDTH:
        scale = INPUT_WIDTH / float(max(w, h))
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

    # Realizar inferencia (devuelve resultados)
    # uso de model.predict está disponible, pero con API simple use model(frame, imgsz=... )
    start = time.time()
    results = model(frame, imgsz=INPUT_WIDTH, conf=CONF_THRESH, verbose=False)
    elapsed = time.time() - start

    # results[0].boxes contiene las cajas, .cls son las clases (float)
    boxes = results[0].boxes
    persona_detectada = False

    if boxes is not None and len(boxes) > 0:
        # cada box tiene .cls y .conf
        for b in boxes:
            cls = int(b.cls[0]) if hasattr(b, "cls") else int(b[5])
            conf = float(b.conf[0]) if hasattr(b, "conf") else float(b[4])
            # Clase COCO para 'person' es 0
            if cls == 0 and conf >= CONF_THRESH:
                persona_detectada = True
                break

    # Persistencia para evitar parpadeos
    if persona_detectada:
        persistencia = PERSISTENCE_FRAMES
    else:
        persistencia = max(0, persistencia - 1)

    persona_final = persistencia > 0

    # Log básico
    print(f"Detected: {persona_detectada} -> final: {persona_final} (inference: {elapsed*1000:.1f} ms)")

    return jsonify({"persona": bool(persona_final)})

if __name__ == "__main__":
    print("Servidor YOLOv8 listo en http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
