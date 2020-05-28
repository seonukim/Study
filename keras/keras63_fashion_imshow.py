'''
2020.05.28 과제
# Sequential() 모델로 CNN, DNN, LSTM모델을 완성하라
# 하단에 주석으로 acc와 loss 결과를 명시하시오.
'''

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

plt.imshow(x_train[0], 'gray')
plt.show()