from scripts.sanitize_reports import portable


def test_public_paths_are_portable_without_changing_metrics():
    root = r"C:\Users\PrivateName\project"
    original = dict(image=root+r"\data\frame.jpg", accuracy=.9997,
                    nested=[root.replace("\\","/")+"/model.onnx",7,None])
    public = portable(original,root)
    assert public == dict(image=r"data\frame.jpg",accuracy=.9997,nested=["model.onnx",7,None])
    assert "PrivateName" in original["image"]
