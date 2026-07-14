def classify_severity(per_class_pct: dict, detections: list):
    damage_pct = per_class_pct.get("cracks", 0) + per_class_pct.get("spalling", 0)
    num_detections = len(detections)
    avg_conf = (sum(d["confidence"] for d in detections) / num_detections
                if num_detections else 0)

    score = (damage_pct * 2.0) + (num_detections * 1.5) + (avg_conf * 10)

    if score < 10:
        level = "Low"
    elif score < 30:
        level = "Medium"
    else:
        level = "High"

    return {
        "severity_level": level,
        "score": round(score, 2),
        "damage_area_pct": round(damage_pct, 2),
        "num_defects_detected": num_detections,
        "avg_detection_confidence": round(avg_conf, 3),
    }