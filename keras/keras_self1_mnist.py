'''파이썬으로 배우는 딥러닝 교과서 실습 코드'''

# 1. 모듈 임포트
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

# 2. 데이터 구성
(x_train, y_train), (x_test, y_test) = mnist.load_data()        # mnist 데이터 로드

x_train = x_train.reshape(x_train.shape[0], 784)[:6000]
x_test = x_test.reshape(x_test.shape[0], 784)[:1000]
y_train = to_categorical(y_train)[:6000]
y_test = to_categorical(y_test)[:1000]                          # train, test 분할

model = Sequential()
model.add(Dense(256, input_dim = 784))      # 입력 차원 : 784, 아웃풋 노드: 256
model.add(Activation("sigmoid"))            # 활성화 함수 : 시그모이드
model.add(Dense(128))                       # 은닉층 노드 : 128
model.add(Activation("sigmoid"))            # 활성화 함수 : 시그모이드
model.add(Dropout(rate = 0.5))              # 
model.add(Dense(10))
model.add(Activation("softmax"))                                # 모델링

# model.summary()

# 3. 컴파일링 및 모델 적합
sgd = optimizers.SGD(lr = 0.1)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy',
              metrics = ['acc'])

hist = model.fit(x_train, y_train, batch_size = 500,
                 epochs = 5, verbose = 1,
                 validation_data = (x_test, y_test))

# 4. 시각화
plt.plot(hist.history['acc'], label = 'acc', ls = '-', marker = 'o')
plt.plot(hist.history['val_acc'], label = 'val_acc', ls = '-', marker = 'x')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc = 'best')
plt.show()

# 5. 예측
res = model.evaluate(x_test, y_test)
print(res)