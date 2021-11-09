from __future__ import print_function
import cv2
from arcface_pytorch.models import *
import torch
import numpy as np
from arcface_pytorch.config import Config
from torch.nn import DataParallel

'''
人脸特征提取器
'''
class Extractor:
    def __init__(self):
        super(Extractor, self).__init__()

        # 1.创建模型
        self.opt = Config()
        if self.opt.backbone == 'resnet18':
            self.model = resnet_face18(self.opt.use_se)
        elif self.opt.backbone == 'resnet34':
            self.model = resnet34()
        elif self.opt.backbone == 'resnet50':
            self.model = resnet50()

        # 2.加载模型参数
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DataParallel(self.model)
        # load_model(model, opt.test_model_path)
        self.model.load_state_dict(torch.load(self.opt.test_model_path))
        # model.load_state_dict(torch.load(opt.test_model_path, map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, crop_img):

        image = load_image(crop_img)
        if image is None:
            print('read {} error'.format(crop_img))

        data = torch.from_numpy(image)
        data = data.to(self.device)
        output = self.model(data)
        output = output.data.cpu().numpy()

        # fe_1为image本身的512维特征，fe_2为image的左右翻转图片的512维特征
        # 对于每张图片，合并本身512维特征+左右翻转的512维特征，得到一个1024维的特征作为该图片的feature
        fe_1 = output[::2]
        fe_2 = output[1::2]
        feature = np.hstack((fe_1, fe_2))
        # print(feature.shape)


        return feature


def load_image(img):
    image = img.convert("L")
    # image = cv2.imread(img, 0)
    image = np.array(image)
    h, w = image.shape
    x_min = min(h, w)
    image = image[h - x_min: h, 0: w]
    image = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)

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



def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)




if __name__ == '__main__':

    exactor = Extractor()
    path = r"arcface_pytorch/face_data/st/1.jpg"
    exactor(path)