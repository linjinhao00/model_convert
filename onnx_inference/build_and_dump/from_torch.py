import torch 
model = torch.load("whole.pth")

dummy_input = torch.randn(1,3, 224, 224)
torch.onnx.export(model, dummy_input, "whole.onnx")