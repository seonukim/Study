import numpy as np
import cv2
import re
import os
from keras.models import load_model
from class_project import Project

# data
#------------- only one file -------------------
# path = 'D:/data/project/testset/bichon.jpg'
# img = cv2.imread(path)
# img = cv2.resize(img, dsize = (128, 128), interpolation = cv2.INTER_LINEAR)
# x_pred = np.array(img/255).reshape(-1, 128, 128, 3)                 # 픽셀 = 0~255값가짐, 학습을 위해 0~1사이의 소수로 변환

#---------------- load_data -------------------
path = 'D:/data/project/testset' 

x_pred = np.load('./project/project02/data/pred_image02.npy')
                                 # filename
print(x_pred.shape)

# load_model
model = load_model('./project/project02/model_save/best_xception.hdf5')

# predict
prediction = model.predict(x_pred)
number = np.argmax(prediction, axis = 1)
print(len(number))

# 카테고리 불러오기
categories = ['Bichon_frise', 'Border_collie', 'Bulldog',
              'Chihuahua', 'Corgi', 'Dachshund',
              'Golden_retriever', 'Huskey', 'Jindo_dog',
              'Maltese', 'Pug', 'Yorkshire_terrier']


f = open('./project/project02/data/pred_image_name.txt', 'r')
filename = f.readlines()

# filename = ['Jindo_dog']

for i in range(len(number)):
    idex = number[i]
    true = filename[i].replace('\n', '')
    pred = categories[idex]
    print('실제 :', true, '\t예측 견종 :', pred)