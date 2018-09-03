from __future__ import absolute_import
from __future__ import print_function
import os
import random
from cv2 import cv2
import numpy as np
import face_recognition
from imutils import paths
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

def load_data(path, image_size):
    """
    从文件中读取图片，并将图片随机乱序排序
    根据分类文件夹打上标签
    :param path: 需要读取的文件路径
    :param image_size:图像张量
    :return:图片数据和标签
    """
    data = []
    labels = []
    # 获得图像路径并随机选取
    imagePaths = sorted(list(paths.list_images(path)))
    lists = sorted(os.listdir(path + "/"))
    random.seed(42)
    random.shuffle(imagePaths)

    # 在输入图像上循环
    for imagePath in imagePaths:
        # 加载图像
        image = face_recognition.load_image_file(imagePath)
        face_locations = face_recognition.face_locations(image)
        # 得到面部坐标
        top = face_locations[0][0]
        right = face_locations[0][1]
        bottom = face_locations[0][2]
        left = face_locations[0][3]
        # 裁剪出面部
        image = image[top - 20:bottom + 20,left - 20:right + 20]
        # 对图片进行预处理
        image = cv2.resize(image, image_size)
        image = img_to_array(image)
        # 将其存储在数据列表中
        data.append(image)

        # 从图像路径中提取类标签, 并更新
        # labels 列表
        label = int(lists.index(imagePath.split(os.path.sep)[-2]))
        labels.append(label)

    # 将原始像素强度缩放到范围 [0,1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # 将标签从整数转换为向量
    labels = to_categorical(labels, num_classes=len(os.listdir(path)))
    return data, labels