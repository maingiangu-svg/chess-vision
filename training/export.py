"""Export a trusted local checkpoint. Evaluate every output on the same boards."""
import argparse
import json
from pathlib import Path
import platform
import numpy as np
import torch
from training.data import write_json
from vision.square_recognition import CLASSES, SquareRecognizer


def export_squares(weights, output, format):
    recognizer = SquareRecognizer(weights)
    model = recognizer.model.eval()
    torch.manual_seed(42)
    example = torch.randn(64, 3, 48, 48)
    with torch.inference_mode():
        expected = model(example).numpy()
    if format == "onnx":
        import onnxruntime as ort
        torch.onnx.export(model, example, str(output), input_names=["squares"], output_names=["logits"],
                          dynamic_axes={"squares": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"squares": example.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        parity = dict(random_tensor_max_abs_error=float(np.max(np.abs(actual-expected))),
                      note="numerical plumbing check only; not camera accuracy")
    else:
        if platform.system() != "Darwin":
            raise RuntimeError("Core ML export/runtime validation must run on the Mac")
        import coremltools as ct
        traced = torch.jit.trace(model, example)
        converted = ct.convert(traced, inputs=[ct.TensorType(name="squares", shape=example.shape)],
                               convert_to="mlprogram")
        converted.save(str(output))
        actual = next(iter(converted.predict({"squares": example.numpy()}).values()))
        np.testing.assert_allclose(actual, expected, rtol=.05, atol=.01)
        parity = dict(random_tensor_max_abs_error=float(np.max(np.abs(actual-expected))))
    write_json(str(output)+".json", dict(classes=list(CLASSES), input_size=48,
               preprocessing="BGR to RGB, float/255, ImageNet mean/std; input NCHW normalized float32",
               parity=parity, camera_accuracy_verified=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("approach", choices=("squares", "yolo"))
    p.add_argument("weights", type=Path)
    p.add_argument("--format", choices=("onnx", "coreml"), required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.approach == "yolo":
        from ultralytics import YOLO
        print(YOLO(str(args.weights)).export(format=args.format, device="cpu", imgsz=640, half=False))
    else:
        output = args.output or args.weights.with_suffix(".onnx" if args.format == "onnx" else ".mlpackage")
        if output.exists():
            raise FileExistsError(output)
        export_squares(args.weights, output, args.format)


if __name__ == "__main__":
    main()
