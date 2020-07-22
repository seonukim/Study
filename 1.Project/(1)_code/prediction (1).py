import numpy as np
import cv2
import re
import os
from keras.models import load_model

# data
path = 'D:/data/project/testset'
'''
image_w = 112
image_h = 112

x_pred = []
for top, dir, f in os.walk(path): 
    for filename in f:
        img = cv2.imread(top+'/'+filename)
        img = cv2.resize(img, dsize = (image_w, image_h), interpolation = cv2.INTER_LINEAR)
        x_pred.append(img/255)     

x_pred = np.array(x_pred)         
np.save('./project/project02/data/pred_img.npy', x_pred)
print(x_pred.shape)
print('----------- x_pred save complete ---------------')
'''
x_pred = np.load('./project/project02/data/pred_img.npy')

# load_model
model = load_model('project\project02\model_save/best.hdf5')

# predict
prediction = model.predict(x_pred)
number = np.argmax(prediction, axis = 1)

# 카테고리 불러오기
categories = np.load('./project/project02/data/category.npy')

filename = os.listdir(path)

for i in range(len(number)):
    idex = number[i]
    true = filename[i].replace('.jpg', '').replace('.png','')
    pred = categories[idex]
    print('실제 :', true, '\t예측 견종 :', pred)

