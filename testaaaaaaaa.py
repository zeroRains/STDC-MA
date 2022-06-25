import os
from tqdm import tqdm
from random import sample

pwd = f'/home/disk2/ray/datasets/DriveSeg'
img_path = os.path.join(pwd, 'frames')
label_path = os.path.join(pwd, 'labels')
train_path = os.path.join(pwd, 'train.txt')
test_path = os.path.join(pwd, 'test.txt')

dirs_name = os.listdir(img_path)

data = []

for name in tqdm(dirs_name):
    if name == '.DS_Store':
        continue
    path = os.path.join(img_path, name)
    files = os.listdir(path)
    for file_name in files:
        img = os.path.join(path, file_name)
        label = img.replace('frames/', 'labels/')
        if not os.path.exists(img) or not os.path.exists(label):
            print(label)
        data.append(f"{img} {label}\n")

index = sample(range(0, len(data)), len(data))
train = int(len(data) * 0.7)
test = len(data) - train

with open(train_path, 'w+') as f_train:
    with open(test_path, 'w+') as f_test:
        for i in tqdm(range(0, len(data))):
            if i < train:
                f_train.write(data[index[i]])
            else:
                f_test.write(data[index[i]])
