import torch
import os
import cv2
from config import Config
from PIL import Image
from torchvision import transforms
import torchvision.models as md
# from face import FaceNet
from models import *
import numpy as np
# from detect import *
from torch.nn import DataParallel

tf = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
])


def get_featurs(model, img_path):
    
    # for i, img_path in enumerate(test_list):
        # load_image函数有点不一样：
        #   合并image本身+image的左右翻转图片为2张图片
        # 因此，在这里的得到的image.shape=(2, 1, 128, 128)
    image = load_image(img_path)
    if image is None:
        print('read {} error'.format(img_path))


    

    data = torch.from_numpy(image)
    # data = data.to(torch.device("cuda"))
    output = model(data)
    output = output.data.cpu().numpy()

    # fe_1为image本身的512维特征，fe_2为image的左右翻转图片的512维特征
    # 对于每张图片，合并本身512维特征+左右翻转的512维特征，得到一个1024维的特征作为该图片的feature
    fe_1 = output[::2]
    fe_2 = output[1::2]
    feature = np.hstack((fe_1, fe_2))
    # print(feature.shape)

    return feature

def load_image(img_path):
    image = cv2.imread(img_path, 0)

    h, w = image.shape
    x_min = min(h, w)
    image = image[h-x_min: h, 0: w]
    image = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
    if image is None:
        return None
    # image.shape=(128, 128)
    # 合并image+image的左右翻转图片
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    # 最终返回的image.shape=(2, 1, 128, 128)
    return image



opt = Config()
if opt.backbone == 'resnet18':
    model = resnet_face18(opt.use_se)
elif opt.backbone == 'resnet34':
    model = resnet34()
elif opt.backbone == 'resnet50':
    model = resnet50()



# 2.加载模型参数
model = DataParallel(model)
# load_model(model, opt.test_model_path)
model.load_state_dict(torch.load(opt.test_model_path))
# model.load_state_dict(torch.load(opt.test_model_path, map_location="cpu"))
model.to(torch.device("cuda"))

model.eval()

fet_data = {}

for cls in os.listdir(r"arcface_pytorch/face_data"):

    fet_data[cls] = []

    for name in os.listdir(fr"arcface_pytorch/face_dat/{cls}"):
        img_path = fr"arcface_pytorch/face_data/{cls}/{name}"

        fet = get_featurs(model, img_path)
        fet_data[cls].append(fet)

torch.save(fet_data, r"arcface_pytorch/face_data")



