import torch

script_model = torch.jit.load("resnet18_script.pth")
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(script_model, dummy_input, "script.onnx")

# dummy_input = torch.randn(1, 3, 224, 224)
# trace_model = torch.jit.load("resnet18_trace.pth")
# torch.onnx.export(trace_model, dummy_input, "trace.onnx")