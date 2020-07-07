import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
path = 'D:/Study/data/dog_cat/'
os.chdir(path)

## 이미지 불러오기
img_dog = load_img(path + 'dog.jpg', target_size = (224, 224))
img_cat = load_img(path + 'cat.jpg', target_size = (224, 224))
img_suit = load_img(path + 'suit.jpg', target_size = (224, 224))
img_yang = load_img(path + 'yang.jpg', target_size = (224, 224))

## 불러온 이미지 확인
img_list = [img_dog, img_cat, img_suit, img_yang]
for i in img_list:
    plt.imshow(i)
    # plt.show()

## 이미지 array형 변환해주기
arr_dog = img_to_array(img = img_dog)
arr_cat = img_to_array(img = img_cat)
arr_suit = img_to_array(img = img_suit)
arr_yang = img_to_array(img = img_yang)

arr_list = [arr_dog, arr_cat, arr_suit, arr_yang]
for i in arr_list:
    print(f'{i}의 타입 : {type(i)}')
    print('\n'f'{i}의 shape : {i.shape}')
# print(arr_dog)
# print(type(arr_dog))            # <class 'numpy.ndarray'>
# print(arr_dog.shape)            # (224, 224, 3)

## vgg16으로 데이터 전처리해주기
# keras.applications.vgg16.preprocess_input     StandardScaler형식?
# RGB -> BGR
arr_dog = keras.applications.vgg16.preprocess_input(arr_dog)
arr_cat = keras.applications.vgg16.preprocess_input(arr_cat)
arr_suit = keras.applications.vgg16.preprocess_input(arr_suit)
arr_yang = keras.applications.vgg16.preprocess_input(arr_yang)

print(arr_dog)      # RGB -> BGR 맨 앞, 뒤의 순서가 바뀜, 이유? VGG16에서 이 순서로 받아들이기 때문, 의미는 없다


## 이미지 데이터를 하나로 합친다
arr_input = np.stack([arr_dog, arr_cat, arr_suit, arr_yang])            # 데이터를 합쳐준다
print(arr_input.shape)          # (4, 224, 224, 3)


## 모델링
model = VGG16()
probs = model.predict(arr_input)
print(probs)
print(f'probs.shape : {probs.shape}')

## 이미지 결과
# keras.applications.vgg16.decode_predictions
results = keras.applications.vgg16.decode_predictions(probs)
print(type(results))
print(results[0])
print(results[1])
print(results[2])
print(results[3])