from ultralytics import YOLO
import numpy as np

_model_cache = {}

def load_detector(weights_path="models/yolo.pt"):
    if weights_path not in _model_cache:
        _model_cache[weights_path] = YOLO(weights_path)
    return _model_cache[weights_path]

def run_detection(image: np.ndarray, weights_path="models/yolo26n.pt",
                   conf=0.25, iou=0.45):
    model = load_detector(weights_path)
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False)
    r = results[0]

    detections = []
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_name": cls_name,
                "confidence": confidence,
                "box": (x1, y1, x2, y2)
            })

    annotated_bgr = r.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    return annotated_rgb, detections