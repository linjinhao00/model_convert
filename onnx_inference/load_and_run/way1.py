import onnxruntime
import cv2
import numpy as np

input_img = cv2.imread("./face.png").astype(np.float32)
input_img = cv2.resize(input_img, (224, 224))
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)
print(input_img.shape)

session = onnxruntime.InferenceSession("./vgg16.onnx")
input = {'input.1': input_img}
output = session.run(['70'], input)[0]
print(output)