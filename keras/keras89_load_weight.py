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

# 모델 구성
model = Sequential()
model.add(Conv2D(256, (2, 2), input_shape = (28, 28, 1), activation = 'relu'))
model.add(Conv2D(112, (2, 2), padding = 'same'))
model.add(Conv2D(64, (2, 2), padding = 'same'))
# model.add(Conv2D(89, (2, 2)))
# model.add(Conv2D(36, (2, 2)))
# model.add(Conv2D(69, (2, 2)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Flatten())
model.add(Dense(23))
model.add(Dense(10, activation = 'softmax'))
# model.add(Dense(10, activation = 'softmax'))

model.summary()

# model.save('./model/model_test01.h5')


# 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# hist = model.fit(x_train, y_train, epochs = 5, batch_size = 100, validation_split = 0.01, callbacks = [es])

# model.save('./model/model_test01.h5')
# model.save_weights('./model/test_weight1.h5')
from keras.models import load_model
# model = load_model('./model/model_test01.h5')       # 가중치는 가져오지 않는다?
model.load_weights('./model/test_weight1.h5')         # 순수하게 weight만 저장됨, layer가 추가되면 에러 발생

# 평가 및 예측
loss_acc = model.evaluate(x_test, y_test)
print("loss, acc : ", loss_acc)

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

'''
acc :  [0.94580805, 0.97905725, 0.9830471, 0.9861616, 0.98777777]   
val_acc :  [0.9750000238418579, 0.9850000143051147, 0.9833333492279053, 0.9816666841506958, 0.9816666841506958]
loss_acc :  [0.06516363149937825, 0.9825000166893005]
'''

'''
acc :  [0.9469192, 0.9803367, 0.9833165, 0.9853367, 0.98779464]     
val_acc :  [0.9766666889190674, 0.9800000190734863, 0.9850000143051147, 0.9800000190734863, 0.9866666793823242]
loss_acc :  [0.0635595706749009, 0.9824000000953674]
'''