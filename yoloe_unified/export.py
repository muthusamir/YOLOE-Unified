import torch
from yoloe_unified.models import YOLOEUnified

def export_to_tensorrt(weights_path, output_path="yoloe_unified.engine"):
    model = YOLOEUnified(weights=weights_path)
    model.eval()

    # Export to ONNX first
    dummy_input = torch.randn(1, 3, 640, 640)
    torch.onnx.export(model, dummy_input, "yoloe_unified.onnx", opset_version=17)

    # Then use trtexec or TensorRT Python API
    print(f"Model exported. Build engine with: trtexec --onnx=yoloe_unified.onnx --fp16 --saveEngine={output_path}")
