import os
import cv2
from tqdm import tqdm

pwd = '/home/disk2/ray/datasets/SUN_RGBD'
train_path = os.path.join(pwd, 'train.txt')
labels_path = os.path.join(pwd, 'test.txt')

with open(train_path, 'r+') as f:
    files = f.readlines()
    for data in tqdm(files):
        data1 = data.strip().split()
        img = data1[0]
        label = data1[1]
        if not os.path.exists(img):
            print(f"{img}  not exist")
            continue
        if not os.path.exists(label):
            print(f"{label}  not exist")
            continue
        img1 = cv2.imread(img)
        label1 = cv2.imread(label)
        if img1.any() == None:
            print(f"{img}  bad")
        if label1.any() == None:
            print(f"{label}  bad")

with open(labels_path, 'r+') as f:
    files = f.readlines()
    for data in tqdm(files):
        data1 = data.strip().split()
        img = data1[0]
        label = data1[1]
        if not os.path.exists(img):
            print(f"{img}  not exist")
            continue
        if not os.path.exists(label):
            print(f"{label}  not exist")
            continue
        img1 = cv2.imread(img)
        label1 = cv2.imread(label)
        if img1.any() == None:
            print(f"{img}  bad")
        if label1.any() == None:
            print(f"{label}  bad")
