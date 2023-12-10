import torch 
model = torch.load("whole.pth")
model.eval()

input_img = cv2.imread("./face.png").astype(np.float32)
input_img = cv2.resize(input_img, (224, 224))
input_img = np.transpose(input_img, [2,0,1])
input_img = np.expand_dims(input_img, 0)
print(input_img.shape)

output = model(torch.from_numpy(input_img)).detach().numpy()