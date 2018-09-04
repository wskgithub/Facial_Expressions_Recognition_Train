import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array

def classify_image(image_lists, kind_lists):
    """
    对测试集图片进行分类，并输出分类日志
    :param model: 读取进来的模型
    :param image_lists:需做分类的图片列表
    :param kind_lists:类型名称列表
    :return:null
    """
    # 从图片列表中遍历每一张需要识别的图片
    image_count = len(image_lists)
    classify_true = 0
    # classfiy_result = False
    model = load_model('models/Base_model.h5')
    for image in image_lists:
        # 将图片送入模型中预测
        result = model.predict(image[0])[0]
        # 取出相似度最高的一项
        proba = np.max(result)
        # 获得识别出类型的标签
        label = kind_lists[int(np.where(result == proba)[0])]
        # 打印分类log
        log = ("result:" + label + " -> " + str(proba * 100) 
            + " -> source:" + image[1] 
            + " -> name:" + image[2] 
            + " -> path:" + image[3]
            + " -> classfiy_result：" + "\n")
        print(log)
        # 判断识别结果是否正确
        if label == image[1]:
            classify_true += 1
            # classfiy_result = True
        else:
            # classfiy_result = False
            # 输出分类错误日志到文件
            with open("log.txt", "a") as f:
                f.write(log)

    print("分类图片总数:{}, 分类正确:{}, 分类正确率:{}%"
        .format(image_count, classify_true, (classify_true/image_count) * 100))