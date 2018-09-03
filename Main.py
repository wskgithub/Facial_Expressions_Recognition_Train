import argparse
import os
import time
from InitializeModel import initialize_model
from LoadData import load_data
from TrainModel import train

FILTERS = 32 # 卷积滤波器数量
IMAGE_SIZE = (64,64) # 图像缩放大小
KERNEL_SIZE = (3,3) # 卷积核大小
INPUT_SHAPE = (64,64,3) # 图像张量
POOL_SIZE = (2,2) # 池化缩小比例因素
NB_CLASSES = 0 # 分类数
EPOCHS = 100 # 循环的次数

# 主函数
if __name__=='__main__':
    print('[INFO] start...')
    train_images_path = "/media/wsk/移动磁盘1/face_images/train"
    test_images_path = "/media/wsk/移动磁盘1/face_images/test"
    # 加载图片
    print("[INFO] loading images...")
    x_train_background, y_train_background = load_data(train_images_path, IMAGE_SIZE)
    x_test_background, y_test_background = load_data(test_images_path, IMAGE_SIZE)
    #初始化模型
    print("[INFO] initialize background model...")
    background_model = initialize_model(FILTERS, KERNEL_SIZE, INPUT_SHAPE, POOL_SIZE,
                                        len(os.listdir(train_images_path)))
    # 训练模型
    print("[INFO] compiling background model...")
    train(background_model, x_train_background, y_train_background, x_test_background, y_test_background,
          8, EPOCHS, 'models/Base_model.h5')

    