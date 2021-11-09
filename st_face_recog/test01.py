import torch

import json
import torchvision
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([])
#
# e = {"c": 3}
#
# # a = json.load(open("44.json", "r"))
# # with open(r"44.json", "w") as f:
# #
# #     json.dump(g, f)
# #     print("ddddd")
#
# a = json.load(open(r"D:\pycharmproject\FaceRecognizition\arcface_pytorch\id_name.json", 'r'))
# print(a)
# for name in a:
#     print(name)

import torch.nn.functional as F
# a = torch.tensor([[1, 5, 2, 3]])
# a = F.normalize(a)

# b = [torch.tensor([1, 5, 2]), a]
import numpy
# b = torch.tensor([[5., 6.]])
# c = torch.tensor([[[5., 6.]], [[6., 9.]]])
# print(b.shape, c.shape)
# b1 = F.normalize(b, dim=1)
# c1 = torch.transpose(F.normalize(c, dim=2), 2,  1)
# print(b1.shape, b1)
# print(c1.shape, c1)
# d = torch.matmul(b1, c1)
# print(d.shape, d)
# b = []
# c = None
# if b:
#     print("aaa")
# else:
#     print("bbb")

# import cv2
# import matplotlib.pyplot as plt
#
#
# img = cv2.imread(r"D:\pycharmproject\FaceRecognizition\sj_82.jpg")
# new_img = cv2.copyMakeBorder(img, 20, 20, 20, 20, borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))
# cv2.imwrite("new_img.jpg", new_img)
#
# import torch
# import torch.nn as nn
#
# def autopad(k, p=None):  # kernel, padding
#     # Pad to 'same'
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p
#
#
# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))
#
#     def fuseforward(self, x):
#         return self.act(self.conv(x))
#
#
# class StemBlock(nn.Module):
#     def __init__(self, c1, c2, k=3, s=2, p=None, g=1, act=True):
#         super(StemBlock, self).__init__()
#         self.stem_1 = Conv(c1, c2, k, s, p, g, act)
#         self.stem_2a = Conv(c2, c2 // 2, 1, 1, 0)
#         self.stem_2b = Conv(c2 // 2, c2, 3, 1, 1)
#         self.stem_2p = nn.MaxPool2d(kernel_size=3,stride=1, padding=1, ceil_mode=True)
#         self.stem_3 = Conv(c2 * 2, c2, 1, 1, 0)
#
#     def forward(self, x):  # x    1, 3, 640, 640
#         stem_1_out  = self.stem_1(x)  # 1, 64, 320, 320
#         stem_2a_out = self.stem_2a(stem_1_out)  # 1, 32, 320, 320
#         stem_2b_out = self.stem_2b(stem_2a_out)  # 1, 64, 160, 160
#         stem_2p_out = self.stem_2p(stem_1_out)   #  # 1, 64, 160, 160
#         out = self.stem_3(torch.cat((stem_2b_out,stem_2p_out),1))
#         return out
#
#
# con = StemBlock(3, 64)
# x = torch.randn(1, 3, 640, 640)
# b = con(x)
# print(b.shape)


from flask import Flask, request, jsonify

import io
from PIL import Image

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"

@app.route("/xx", methods = ["POST"])
def ss():


    name = request.form.get("name")

    file = request.files.get("file")
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image.show()
    return jsonify({"name": name, "filelen": len(image_bytes)})

if __name__ == "__main__":
    app.run()