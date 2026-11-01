"""13-class square adapter sharing TemplateRecognizer's uncertainty contract.

Callers still pass observations through the existing legal-move tracker.
"""
from pathlib import Path
import cv2
import numpy as np
from shared.schemas import PIECE_CLASSES
from vision.template_recognition import square_image

CLASSES = (*PIECE_CLASSES, "empty")
SQUARES = tuple(f"{f}{r}" for r in range(1, 9) for f in "abcdefgh")


def build_model(pretrained=False, architecture="mobilenet_v3_small"):
    import torch
    if architecture == "tiny_cnn":
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, 3, padding=1), torch.nn.BatchNorm2d(24), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(24, 48, 3, padding=1), torch.nn.BatchNorm2d(48), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(48, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Flatten(), torch.nn.Linear(64*6*6, 128), torch.nn.ReLU(), torch.nn.Dropout(.15),
            torch.nn.Linear(128, len(CLASSES)))
    if architecture != "mobilenet_v3_small":
        raise ValueError(f"Unknown architecture: {architecture}")
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    return model


def tensor_from_bgr(patches):
    import torch
    rgb = np.stack([cv2.cvtColor(p, cv2.COLOR_BGR2RGB) for p in patches])
    tensor = torch.from_numpy(rgb.transpose(0, 3, 1, 2).copy()).float()/255
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    return (tensor-mean)/std


class SquareRecognizer:
    def __init__(self, weights, device="cpu", confidence=.9, margin=.25, fallback=None):
        import torch
        if not 0 <= confidence <= 1 or not 0 <= margin <= 1:
            raise ValueError("confidence and margin must be in [0, 1]")
        self.device, self.threshold, self.margin, self.fallback = device, confidence, margin, fallback
        self.backend = Path(weights).suffix
        if self.backend in (".onnx", ".mlpackage"):
            import json
            metadata = json.loads(Path(str(weights)+".json").read_text())
            if self.backend == ".onnx":
                import onnxruntime as ort
                self.model = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
            else:
                import coremltools as ct
                self.model = ct.models.MLModel(str(weights))
        else:
            checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
            metadata = checkpoint
            self.model = build_model(architecture=checkpoint.get("architecture", "mobilenet_v3_small"))
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(device).eval()
        if tuple(metadata["classes"]) != CLASSES or metadata.get("input_size") != 48:
            raise ValueError("checkpoint mapping/preprocessing differs from shared.schemas")

    def probabilities(self, patches):
        import torch
        x = tensor_from_bgr(patches)
        if self.backend == ".onnx":
            logits = torch.from_numpy(self.model.run(None, {"squares": x.numpy()})[0])
        elif self.backend == ".mlpackage":
            logits = torch.from_numpy(next(iter(self.model.predict({"squares": x.numpy()}).values())))
        else:
            with torch.inference_mode():
                logits = self.model(x.to(self.device)).cpu()
        return logits.softmax(1).numpy()

    def predict_labels(self, rectified, orientation):
        patches = [square_image(rectified, s, orientation) for s in SQUARES]
        probabilities = self.probabilities(patches)
        if probabilities.shape != (64, 13) or not np.isfinite(probabilities).all():
            raise ValueError("nonfinite or malformed model output")
        predicted, scores = {}, {}
        for square, row in zip(SQUARES, probabilities):
            ranked = np.argsort(row)
            best, second = ranked[-1], ranked[-2]
            score = float(row[best])
            predicted[square] = CLASSES[best] if score >= self.threshold and score-row[second] >= self.margin else "uncertain"
            scores[square] = score
        return predicted, scores

    def predict(self, rectified, orientation):
        labels, scores = self.predict_labels(rectified, orientation)
        errors = [s for s, label in labels.items() if label == "uncertain"]
        if errors and self.fallback is not None:
            return self.fallback.predict(rectified, orientation)
        occupied = {s: label for s, label in labels.items() if label not in ("empty", "uncertain")}
        return occupied, min(scores.values()), errors
