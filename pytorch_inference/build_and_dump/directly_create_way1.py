import torchvision.models as models

# 方式1 ： 加载现成的模型
model = models.vgg16(pretrained=True)

torch.save(model.state_dict(), 'only_weights.pth')
torch.save(model, 'whole.pth')