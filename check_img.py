import os
import cv2
from tqdm import tqdm
import numpy as np

pwd = '/home/disk2/ray/datasets'
all_path = os.path.join(pwd, 'ADE20K_2021_17_01/all.txt')
label_pixel_path = os.path.join(pwd, 'ADE20K_2021_17_01/label_class.txt')
labels_path = os.path.join(pwd, 'ADE20K_2021_17_01/labels')

pixel = []

## 检查所有图像，并获取所有像素
with open(all_path, 'r+') as f:
    files = f.readlines()
    print("获取像素中")
    for data in tqdm(files):
        data1 = data.strip().split()
        img = data1[0]
        label = data1[1]
        if not os.path.exists(img):
            print(f"{img}  not exist")
            continue
        label1 = cv2.imread(label)
        h, w, c = label1.shape
        pixel.extend(list(set([tuple(t) for i in label1 for t in i])))

pixel = list(set(pixel))
pixel = [f'{t[0]}-{t[1]}-{t[2]}' for t in pixel]
print(pixel)
print(len(pixel))
mapping = {}
## 获取类别对应关系
with open(label_pixel_path, 'w+') as f:
    for i in range(len(pixel)):
        mapping[pixel[i]] = i
        f.write(f"{i}-{pixel[i]}\n")

## 生成新的标签
with open(all_path, 'r+') as f:
    files = f.readlines()
    print("生成新标签中")
    for data in tqdm(files):
        file_name = data.strip().split('/')[-1]
        data1 = data.strip().split()
        label = data1[1]
        if not os.path.exists(label):
            print(f"{label}  not exist")
            continue
        label1 = cv2.imread(label)
        # if label1.any() == None:
        #     print(f"{label}  bad")
        h, w, c = label1.shape
        new_label = np.zeros((h, w))
        new_label_path = os.path.join(labels_path, file_name)
        for i in range(h):
            for j in range(w):
                tmp = label1[i, j, :].reshape(-1)
                str = f"{tmp[0]}-{tmp[1]}-{tmp[2]}"
                new_label[i][j] = mapping[str]
        cv2.imwrite(new_label_path, new_label)
