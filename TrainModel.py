from __future__ import absolute_import
from __future__ import print_function
import os
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from LoadData import load_data
# from ResetModel import reset_model

def train(model, x_train, y_train, x_test, y_test, batch_size, epochs, model_name):
    """
    训练模型
    对训练的图像做图像增强
    :param model:需训练的模型
    :param x_train:训练数据
    :param y_train:训练数据标签
    :param x_test:测试数据
    :param y_test:测试数据标签
    :param batch_size:一批数据的大小
    :param epochs:循环的次数
    :param model_name:模型名称
    :return:null
    """
    # 使用TensorBoard对训练过程进行可视化
    """
    tb = TensorBoard(log_dir='./logs',  # log 目录
                     histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     batch_size=32,  # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=False,  # 是否可视化梯度直方图
                     write_images=False,  # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)
    callbacks = [tb]
    """
    # 配置训练模型
    model.compile(optimizer='rmsprop', metrics=['accuracy'], loss='categorical_crossentropy')
    # 图像增强，左右随机移动10%像素
    data_gen = ImageDataGenerator(rotation_range = 30)
    # 逐批生成数据训练模型
     # model.fit_generator(data_gen.flow(x_train, y_train, batch_size=batch_size),
     #                   steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
     #                   verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)
    model.fit_generator(data_gen.flow(x_train, y_train, batch_size=batch_size),
                       steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
                       verbose=1, validation_data=(x_test, y_test))
    # 训练结束保存模型
    print("[INFO] save model...")
    model.save(model_name)
