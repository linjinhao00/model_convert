import torch
from torchvision.models import resnet18

model = resnet18()
model.eval()

# torch.jit.script
with torch.no_grad():
    script_model = torch.jit.script(model)
    print(script_model.graph)
torch.jit.save(script_model, "resnet18_script.pth")