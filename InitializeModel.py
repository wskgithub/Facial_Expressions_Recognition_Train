from __future__ import absolute_import
from __future__ import print_function
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential


def initialize_model(filters, kernel_size, input_shape, pool_size, nb_classes):
    """
    初始化模型，构建卷积成和全连接层
    :param filters:卷积滤波器数量
    :param kernel_size:卷积核大小
    :param input_shape:图像张量
    :param pool_size:池化缩小比例因素
    :param nb_classes:分类数
    :return:初始化后的CNN模型
    """
    # 生成模型
    model = Sequential()

    #####特征层#####
    # 第一个卷积层
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu',
                     input_shape=input_shape))
    # 池化
    model.add(MaxPooling2D(pool_size=pool_size))
    # 第二个卷积层
    model.add(Conv2D(filters=filters*2, kernel_size=kernel_size, activation='relu'))
    # 池化
    model.add(MaxPooling2D(pool_size=pool_size))
    # 第三个卷积层
    model.add(Conv2D(filters=filters*2*2, kernel_size=kernel_size, activation='relu'))
    # 第四个卷积层
    # model.add(Conv2D(filters=filters*2*2*2, kernel_size=kernel_size, activation='relu'))
    #池化
    model.add(MaxPooling2D(pool_size=pool_size))
    #####全链接层#####
    # 压缩维度
    model.add(Flatten())
    # 全链接层
    model.add(Dense(128, activation='relu'))
    # 模型平均，防止过拟合
    model.add(Dropout(0.5))
    # Softmax分类
    model.add(Dense(nb_classes, activation='softmax'))
    return model