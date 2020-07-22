import numpy as np
import cv2
import glob
import os
import re

path = 'D:/data/project/breed'      # dataset 상위 폴더 경로

# 카테고리
def categories(top_folder_path):             # folder이름으로 카테고리 생성
    category = os.listdir(top_folder_path)
    print(category)
    return category

cate = categories(path)
np.save('./project/project02/data/category.npy', cate)
print('--------- categories save complete -------------')


# 이미지 불러오기
def load_image_label(path, w, h):   # 폴더별로 이미지 불러오고 labeling
    groups_floder_path = path
    folder_name = os.listdir(path) 
    num_classes = len(folder_name)               # 카테고리 갯수

    # resize
    image_w = w     # width
    image_h = h     # height

    X = []
    Y = []

    for idex, folder in enumerate(folder_name):
        # one-hot encoding
        label = [0 for i in range(num_classes)]             # [0 0 0 .... 0 0]    
        label[idex] = 1                                     # 해당 자리에 1을 채워 넣음 -> 원핫 인코딩
        image_dir = groups_floder_path + '/'+ folder + '/'  

        for top, dir, f  in os.walk(image_dir):
            # for filename, i in zip(f, range(101)): 
            for filename in f:  
                print(image_dir+filename)
                img = cv2.imread(image_dir + filename)
                img = cv2.resize(img, dsize = (image_w, image_h), interpolation = cv2.INTER_LINEAR)
                X.append(img/255)               # 픽셀 = 0~255값가짐, 학습을 위해 0~1사이의 소수로 변환
                Y.append(label)
                # if i == 100:                  # 이미지 100개만 가져오기
                #     break

    return np.array(X), np.array(Y)

X, Y = load_image_label(path, 112, 112)
print(X.shape)
print(Y.shape)                                  

np.save('./project/project02/data/dog_image_2.npy', X)
np.save('./project/project02/data/dog_label_2.npy', Y)
print('-----Data Save Complete------')
