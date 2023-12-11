import torch
from torchvision.models import resnet18

model = resnet18()
model.eval()

# torch.jit.trace
dummy_input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    trace_model = torch.jit.trace(model, dummy_input)

torch.jit.save(trace_model, "resnet18_trace.pth")
