# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import cv2
from models import *
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path, 0)
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


def get_featurs(model, test_list, batch_size=10):
    '''
    根据batch_size去做模型的前向推理，
    并将所有的结果feature都vstack到features中，
    最后返回features和迭代次数cnt。
    '''
    images = None
    features = None
    cnt = 0  # 迭代轮次
    for i, img_path in enumerate(test_list):
        # load_image函数有点不一样：
        #   合并image本身+image的左右翻转图片为2张图片
        # 因此，在这里的得到的image.shape=(2, 1, 128, 128)
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            output = output.data.cpu().numpy()

            # fe_1为image本身的512维特征，fe_2为image的左右翻转图片的512维特征
            # 对于每张图片，合并本身512维特征+左右翻转的512维特征，得到一个1024维的特征作为该图片的feature
            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    '''
    遍历每个余弦相似度，都将阈值，计算该阈值下的acc精度，
    如果acc是优的，就保留该精度best_acc和阈值best_th
    '''
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    # 通过模型提取"所有图片"的features
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    # 构造特征字典，key为图片的路径名，value为图片对应的feature
    fe_dict = get_feature_dict(identity_list, features)
    # 所有图片的feature都保存在fe_dict字典中，此时就可以计算精度和最佳阈值：
    # 1）直接读取lfw_test_list文件中每一行数据，计算改行两张图片的余弦相似度
    # 2）暴力筛选最佳阈值和最佳精度：
    #    遍历所有余弦相似度作为阈值，计算该阈值下的精度，选择精度最高的那个作为最佳阈值。
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


if __name__ == '__main__':

    # 1.创建模型
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

    # 3.将lfw_test_pair.txt中的所有图片路径添加到identity_list中———已去重，不会添加相同的图片
    # img_paths是图片的绝对路径列表
    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    # 4.测试
    # 1）提取测试集所有图片的特征features
    # 2）计算测试文件lfw_test_list中每一行两张图片的余弦相似度
    # 3）暴力筛选得到最佳阈值best_th和最佳精度best_acc：
    #     遍历所有的余弦相似度将其作为阈值，计算在该阈值下的精度，选取acc最高的对应阈值作为best_th。
    model.eval()
    lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)




