import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

data_path = "/home/disk2/ray/datasets/PascalVocSeg"
train_path = os.path.join(data_path, 'train')
train_labels_path = os.path.join(data_path, 'train_labels')
val_path = os.path.join(data_path, 'val')
val_labels_path = os.path.join(data_path, 'val_labels')
new_label_path = os.path.join(data_path, 'SegmentationClassPixel')

if not os.path.exists(new_label_path):
    os.mkdir(new_label_path)


def transform_label_type(path):
    print(path)
    # 将有色转成二值图像
    labels = os.listdir(path)
    for name in tqdm(labels):
        img = cv2.imread(os.path.join(path, name))
        label = np.zeros(img.shape[0:-1])
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                color_bgr = str(img[i][j][0]) + '-' + str(img[i][j][1]) + '-' + str(img[i][j][2])
                # print(color_bgr)
                if color_bgr not in class_map.keys():
                    label[i][j] = 255
                else:
                    label[i][j] = class_map[color_bgr]['label']
        cv2.imwrite(os.path.join(new_label_path, name), label)


def generator_file(path, name_file):
    print(name_file)
    file_list = os.listdir(path)
    with open(name_file, 'w+') as f:
        for name in tqdm(file_list):
            img = os.path.join(path, name)
            label = os.path.join(new_label_path, name.replace(".jpg", '.png'))
            f.write(f"{img} {label}\n")


# 获取类别
class_dict_path = os.path.join(data_path, 'class_dict.csv')
class_dict = pd.read_csv(class_dict_path)
class_map = {}
for i in range(class_dict.shape[0]):
    color_bgr = str(class_dict.loc[i, 'b']) + "-" + str(class_dict.loc[i, 'g']) + "-" + str(class_dict.loc[i, 'r'])
    class_map[color_bgr] = {"name": class_dict.loc[i, 'name'], "label": i}

# 生成训练文件
generator_file(train_path, os.path.join(data_path, "train_seg.txt"))
generator_file(val_path, os.path.join(data_path, "val_seg.txt"))

# 将伪彩色图像转化为灰度图像
transform_label_type(val_labels_path)
transform_label_type(train_labels_path)
