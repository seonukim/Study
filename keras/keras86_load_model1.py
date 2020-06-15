import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler      # (x - 최소) / (최대 - 최소)
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
print("y_train : ", y_train[0])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(x_train[0].shape)
# plt.imshow(x_train[0], 'gray')          # plt.imshow() 함수는 데이터의 이미지를 보여준다.
# plt.imshow(x_train[0])
# plt.show()

# 데이터 전처리 - 원핫인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)

# 데이터 전처리 - 정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.0
'''
1. 데이터를 CNN 모델에 넣기 위해 reshape해서 4차원으로 바꿔준다. (채널 1 추가)
2. astype('float32')는 현재 정수형인 데이터를 실수형으로 바꿔준다.
3. / 255는 정규화를 의미한다. (MinMaxScaler와 거의 동일)
'''

from keras.models import load_model
model = load_model('./model/model_test01.h5')

model.summary()

# compile, fit
model.compile(loss = 'categorical_crossentropy',
              metrics = ['acc'],
              optimizer = 'adam')
model.fit(x_train, y_train, epochs = 5, batch_size = 32)

# 평가 및 예측
loss_acc = model.evaluate(x_test, y_test)
print("loss, acc : ", loss_acc2


# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['accuracy']
# val_acc = hist.history['val_accuracy']

# print('acc : ', acc)
# print('val_acc : ', val_acc)
# print('loss_acc : ', loss_acc)


# import matplotlib.pyplot as plt

# plt.figure(figsize = (10, 6))               # 그래프의 크기를 (10, 6) 인치로

# plt.subplot(2, 1, 1)                        # 2행 1열의 그래프 중 첫번째 그래프

# '''x축은 epoch로 자동 인식하기 때문에 y값만 넣어준다.'''

# plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')              
# plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
# plt.grid()                                  # 바탕에 격자무늬 추가
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc = 'upper right')

# plt.subplot(2, 1, 2)                        # 2행 1열의 두번째 그래프
# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.grid()                                  # 바탕에 격자무늬 추가
# plt.title('accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['accuracy', 'val_accuracy'])

# plt.show()
