import time
import cv2
import torch
import json
import zmq

import argparse
from milvus_engine import MilvusEngine
import numpy as np
from PIL import Image
from arcface_pytorch import face_extractor
from yolov5_face_master import face_bboxes
import torch.nn.functional as F
import sys
import os
sys.path.append(r"yolov5_face_master")

# 余弦相似度
def compare(face1, face2):
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    cosa = torch.matmul(face1_norm, face2_norm.T)
    return cosa

'''
实现了人脸的注册和人脸识别
'''
class FaceRecongnization:
    def __init__(self, detect_best_weight, save_img_dir, feature_org, use_milvus):
        super(FaceRecongnization, self).__init__()

        # 初始化人脸检测器
        self.detector      = face_bboxes.Detector(detect_best_weight)

        # 初始化特征提取器
        self.extractor     = face_extractor.Extractor()

        # 在人脸注册时同时存储对应人脸照片的地址
        self.save_img_dir  = save_img_dir
        self.device        = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 本地存储特征的地址
        self.feature_org   = feature_org
        self.read_json     = json.load(open(feature_org, "r"))

        # 使用milvus
        self.use_milvus    = use_milvus
        # self.milvus        = MilvusEngine()
        self.milvus        = None

    def Register(self, url, name):

        # 获取视频输入裁剪的图片
        crop_img_list = self.crop_img_from_video(url)

        # 保存图片到指定位置
        self.save_img_from_video(crop_img_list, name)

        if self.use_milvus:

            feature_milvus_list = self.register_to_milvus_extractor(crop_img_list)
            self.milvus.save(name, feature_milvus_list)
        else:
            # 获取特征向量的list存入本地json
            feature_list = self.register_to_json_extractor(crop_img_list)

            # 将特征向量和对应的人名存入字典中
            self.read_json[name] = feature_list

            # 保存在本地json中
            json.dump(self.read_json, open(self.feature_org, "w"))

    def register_to_milvus_extractor(self, crop_img_list):

        feature_milvus_list = []
        for image in crop_img_list:

            feature = self.extractor(image)
            feature_milvus_list.append(feature.tolist())

        # milvus中要存归一化后的特征向量
        feature_milvus_list = F.normalize(torch.tensor(feature_milvus_list).squeeze(1))
        # milvus不支持torch
        feature_milvus_list = feature_milvus_list.numpy()
        return feature_milvus_list

    def register_to_json_extractor(self, crop_img_list):

        feature_list = []
        for image in crop_img_list:

            feature = self.extractor(image)
            feature_list.append(feature.tolist())
        return feature_list

    def save_img_from_video(self, crop_img_list, name):
        '''
        方便检查resigter的人脸是否正确
        '''

        # 创建name目录
        name_dir = os.path.join(self.save_img_dir, name)
        if not os.path.exists(name_dir):
            os.mkdir(name_dir)

        # 保存图片
        for i, crop_img in enumerate(crop_img_list):
            crop_img_path = os.path.join(name_dir, f"{i + 1}.jpg")
            crop_img.save(crop_img_path)

    def crop_img_from_video(self, url):
        cap = cv2.VideoCapture(url, cv2.CAP_DSHOW)

        counter = 0
        crop_img_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            counter += 1
            bboxes = self.detector(frame.copy())
            for box in bboxes:
                if counter % 6 == 0:
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                    M = self.Matrix_face_alignment(box)
                    alignment_image = self.to_alignment(M, frame.copy())

                    crop_img = alignment_image[..., ::-1]
                    crop_img = Image.fromarray(crop_img)
                    crop_img_list.append(crop_img)

            cv2.imshow("1", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return crop_img_list

    def get_data_from_milvus(self):
        data_dict_list = []

        collections = self.milvus.list()
        for collection_name in collections:
            data_dict = {"name": collection_name, "features": []}
            data_dict_list.append(data_dict)
        return data_dict_list

    def VideoFace(self, url):
        '''
        检测视频人脸并进行识别
        '''
        # context = zmq.Context()
        # socket = context.socket(zmq.REP)
        # socket.bind("tcp://*:11556")

        cap = cv2.VideoCapture(url, cv2.CAP_DSHOW)  # 创建视频流采集对象VideoCapture

        while True:  # 循环读取视频流，检测人脸并进行识别，实时显示识别结果
            ret, frame = cap.read()  # 按帧读取视频，ret,frame是获cap.read()方法的两个返回值。其中ret是布尔值，如果读取帧是正确的则返回True，
                                     # 如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
            if not ret:
                continue
            t0 = time.time()
            bboxes = self.detector(frame)  # 检测器会返回这一帧图片中包含的总共的人脸框
            t1 = time.time()
            print(f"检测器检测一张图片需要------>  {t1 - t0}s")
            if bboxes:  # 判断是否有框
                for box in bboxes:
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    M = self.Matrix_face_alignment(box)  # 获得人脸对齐的仿射变换矩阵
                    alignment_image = self.to_alignment(M, frame.copy())  # 通过cv2.warpAffine()仿射变换得到对齐后标准的人脸，并且大小为(128*128)
                    crop_img = Image.fromarray(alignment_image)
                    feature_test = self.extractor(crop_img)  # 特征提取器返回每个框对应的(1, 1024)维特征向量


                    if self.use_milvus:  # 是否使用milvus检索向量
                        cls_list = []
                        max_score = []
                        feature_milvus_tset = F.normalize(torch.tensor(feature_test)).numpy()  # milvus不支持torch.tensor

                        t2 = time.time()
                        for dict in self.get_data_from_milvus():
                            name = dict['name']
                            compare_score = self.milvus.search(feature_milvus_tset, name)
                            # print(compare_score)
                            cls_score = []
                            cls_list.append(name)
                            cls_score.append(compare_score)
                            # 每个存储类别取出网络认为每个框中最大的余弦相似度值
                            max_score.append(compare_score)
                        t3 = time.time()
                        print(f"milvus 需要 {t3 - t2}s")
                    else:
                        cls_list = []
                        max_score = []

                        t4 = time.time()
                        for cls in self.read_json:

                            cls_score = []
                            cls_list.append(cls)
                            for fet in self.read_json[cls]:
                                # 将框和存储的每个人的多个特征一一对比
                                score = compare(torch.tensor(feature_test), torch.tensor(fet))
                                cls_score.append(score)
                            # print(cls_score)
                            # 每个存储类别取出网络认为每个框中最大的余弦相似度值
                            max_score.append(max(cls_score))
                        t5 = time.time()
                        print(f"使用本地向量检索需要---->{t5 - t4}s")
                    if max(max_score) > 0.4:
                        print(f"最大余弦相似度是---->{max(max_score)}")
                        index = np.argmax(max_score)
                        # print(index)
                        pred_cls = cls_list[index]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, pred_cls, (x1, y1-2), 0, 3, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
                cv2.imshow("1", frame)
                t_end = time.time()
                print(f"识别一帧图片需要---->{t_end - t0}s \n")
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            else:  # 无检测框则继续show
                cv2.imshow("1", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # zmq_server
            image_data = cv2.imencode(".jpg", frame)[1].tobytes()
            # print(f"Wait message, image_data = {len(image_data) / 1024:.2f} KB")
            # message = socket.recv()
            # print("message = ", message)
            # socket.send(image_data)

        # cap.release()
        # cv2.destroyAllWindows()

    def ImageRecognize(self, url):
        '''
        检测图片人脸并进行识别
        '''

        image = cv2.imread(url)
        bboxes = self.detector(image)  # 检测器会返回这一帧图片中包含的总共的人脸框

        if bboxes:  # 判断是否有框
            for box in bboxes:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                M = self.Matrix_face_alignment(box)
                alignment_image = self.to_alignment(M, image.copy())
                crop_img = Image.fromarray(alignment_image)  # 只有PIL才能使用crop进行图片的裁剪
                feature_test = self.extractor(crop_img)  # 特征提取器返回每个框对应的(1, 1024)维特征向量

                if self.use_milvus:  # 是否使用milvus检索向量
                    cls_list = []
                    max_score = []
                    feature_milvus_tset = F.normalize(torch.tensor(feature_test)).numpy()  # milvus不支持torch.tensor

                    for dict in self.get_data_from_milvus():
                        name = dict['name']
                        compare_score = self.milvus.search(feature_milvus_tset, name)
                        print(compare_score)
                        cls_score = []
                        cls_list.append(name)
                        cls_score.append(compare_score)
                        # 每个存储类别取出网络认为每个框中最大的余弦相似度值
                        max_score.append(compare_score)

                else:
                    cls_list = []
                    max_score = []
                    for cls in self.read_json:

                        cls_score = []
                        cls_list.append(cls)
                        for fet in self.read_json[cls]:
                            # 将框和存储的每个人的多个特征一一对比
                            score = compare(torch.tensor(feature_test), torch.tensor(fet))
                            # print(score)
                            cls_score.append(score)
                        # print(cls_score)
                        # 每个存储类别取出网络认为每个框中最大的余弦相似度值
                        max_score.append(max(cls_score))

                if max(max_score) > 0.4:
                    print(max(max_score))
                    index = np.argmax(max_score)
                    # print(index)
                    pred_cls = cls_list[index]

                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, pred_cls, (x1, y1-2), 0, 3, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)

            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #     plt.imshow(image[..., ::-1])
        #     plt.show()
        # else:
        #     plt.imshow(image[..., ::-1])
        #     plt.show()

    def FaceAlignment(self, landmarks, std_landmarks):

        # 构建Q矩阵，10*4
        Q = np.zeros((10, 4))

        # 构建S矩阵，10*1
        S = std_landmarks.reshape(-1, 1)

        for i in range(5):
            x, y = landmarks[i]
            Q[i * 2 + 0] = x, y, 1, 0
            Q[i * 2 + 1] = y, -x, 0, 1

        M = (np.linalg.inv(Q.T @ Q) @ Q.T @ S).squeeze()
        # np.linalg.lstsq(Q, S)[0]

        # M[0] = cos t * s
        # M[1] = sin t
        # M[2] = ox
        # M[3] = oy
        # matrix = [
        #     cos t * s,  sin t,  ox
        #     -sin t, cos t * s,  oy
        # ]
        matrix = np.array([
            [M[0], M[1], M[2]],
            [-M[1], M[0], M[3]]
        ])

        return matrix

    def Matrix_face_alignment(self, box):

        std_landmarks = np.array([  # 通过对arcface中作者对齐后的人脸，标注10张，获得通用的标准化的关键点坐标
            [40.979567061066064, 41.516426774567584],
            [88.8882824975239, 40.93974246129967],
            [66.18679549114333, 67.77924169228517],
            [44.374469330991076, 89.78187673839847],
            [86.25706338749815, 90.07465963987703]
        ], dtype=np.float32)

        landmarks = np.array([
            [box[4], box[5]],
            [box[6], box[7]],
            [box[8], box[9]],
            [box[10], box[11]],
            [box[12], box[13]]
        ], dtype=np.float32)

        Matrix = self.FaceAlignment(landmarks, std_landmarks)
        return Matrix

    def to_alignment(self, M, image):

        alignment_image = cv2.warpAffine(image, M, dsize=(128, 128))  # 128*128是我们特征提取器需要的尺寸
        return alignment_image



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect_best_weight", type=str, default=
        r"yolov5_face_master/runs/train/exp5/weights/best.pt", help='detect weight')
    parser.add_argument("--feature_org", type=str, default=r"arcface_pytorch/id_name.json", help='save feature for local dir')
    parser.add_argument("--save_crop_dir", type=str, default=r"arcface_pytorch/face_data", help='save face for local dir')
    parser.add_argument("--url", type=int, default=0, help='recognize Imagae')
    parser.add_argument("--use_milvus", action="store_true", help="database use milvus")
    parser.add_argument("--milvus", action="store_true", help="milvus open")
    parser.add_argument("--name", type=str, help="register name")
    args = parser.parse_args()

    # 启动人脸识别
    face = FaceRecongnization(args.detect_best_weight, args.save_crop_dir, args.feature_org, args.use_milvus)

    # 人脸注册
    # face.Register(args.url, args.name)

    # 视频人脸识别
    face.VideoFace(args.url)

    # args.url = r"./faces.jpg"
    # # 识别图片
    # face.ImageRecognize(args.url)
