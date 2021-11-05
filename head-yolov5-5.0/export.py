
import torch
# import cv2
# import torchvision.models as models
# import torchvision.transforms as T

import torch.onnx

weight = "/datav/shared/face_recognize/13_st/yolov5-5.0/weights/yolov5s.pt"
# weight = "/datav/shared/face_recognize/13_st/yolov5-5.0/runs/train/4level/weights/best.pt"
checkpoint = torch.load(weight, map_location="cpu")
model = checkpoint["model"].float()
model.eval()
model.fuse()
model.model[-1].export = False

input = torch.zeros(1, 3, 640, 640)


torch.onnx.export(model, (input,), "yolov5s.onnx", 
    input_names=["images"], 
    output_names=["output"],
    opset_version=11,
    verbose=False)
