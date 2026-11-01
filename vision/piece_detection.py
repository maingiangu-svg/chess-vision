from pathlib import Path
from shared.schemas import Detection, PIECE_CLASSES

class YoloPieceDetector:
    """Optional Ultralytics adapter. Weights are intentionally supplied by the user."""
    def __init__(self, weights: str):
        if not Path(weights).exists(): raise FileNotFoundError(weights)
        try:
            from ultralytics import YOLO
        except ImportError as e: raise RuntimeError("Install optional dependency: pip install -e '.[vision]'") from e
        self.model = YOLO(weights)
        labels = {str(v).lower().replace("-", "_") for v in self.model.names.values()}
        missing = set(PIECE_CLASSES) - labels
        if missing: raise ValueError(f"Model lacks chess classes: {', '.join(sorted(missing))}")
    def predict(self, image, confidence: float = .35) -> list[Detection]:
        result = self.model.predict(image, conf=confidence, verbose=False)[0]
        names = result.names
        detections = []
        for box, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            label = str(names[int(cls)]).lower().replace("-", "_")
            if label not in PIECE_CLASSES: continue
            detections.append(Detection(label, float(score), tuple(map(float, box))))
        return detections
