import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

## 데이터 불러오기
x_data, y_data = load_iris(return_X_y = True)
print(f'x_data : {x_data.shape} \ny_data : {y_data.shape}')
'''
x_data : (150, 4)
y_data : (150,)
'''

## 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data,
    test_size = 0.2, shuffle = True)

# 텐서플로의 원핫인코딩
# aaa = tf.one_hot(y, ???)
