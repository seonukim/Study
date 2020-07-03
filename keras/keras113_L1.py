# regularizer
'''
L1 규제 : 가중치의 절대값 합
regularizer.l1(1 = 0.01)

L2 규제 : 가중치의 제곱 합
regularizer.l2(1 = 0.01)

loss = L1 * reduce_sum(abs(x))
loss = L2 * reduce_sum(square(x))

다음 레이어로 전달되는 loss 값을 축소함
가중치를 규제하며, 모델 복잡도에 penalty를 준다
'''

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten
from keras.layers import MaxPooling2D, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2                          # l1, l2
                                                                    # l1_l2 : 두 개 같이 쓸 수 있음
modelfath = './model/cifar100_{epoch:02d} - {val_loss:.4f}.hdf5'

# 클래스 객체 생성
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
cp = ModelCheckpoint(filepath = modelfath, monitor = 'val_loss',
                     mode = 'auto', save_best_only = True)
# tb_hist = TensorBoard(log_dir = './graph', histogram_freq = 0,
#                       write_graph = True, write_images = True)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)            # (50000, 32, 32, 3)
print(x_test.shape)             # (10000, 32, 32, 3)
print(y_train.shape)            # (50000, 1)
print(y_test.shape)             # (10000, 1)

# 1-1. 정규화
x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.0
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.0

# 1-2. OHE
# y_train = np_utils.to_categorical(y_train, num_classes = 100)
# y_test = np_utils.to_categorical(y_test, num_classes = 100)
# print(y_train.shape)
# print(y_test.shape)


# 2. 모델링

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                 input_shape = (32, 32, 3), padding = 'same',
                 activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), 
                 padding = 'same', activation = 'relu',
                 kernel_regularizer = l2(0.001)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu',
                 kernel_regularizer = l2(0.001)))
model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu',
                 kernel_regularizer = l2(0.001)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 128, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu',
                 kernel_regularizer = l2(0.001)))
model.add(Conv2D(filters = 128, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu',
                 kernel_regularizer = l2(0.001)))               # hidden layer의 넣고 싶은 곳에 넣어준다.
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()


# 3. 컴파일 및 훈련
model.compile(loss = 'sparse_categorical_crossentropy',         # 원핫인코딩을 하지 않았을 때, 다중분류 손실함수
              metrics = ['accuracy'],                           # sparse는 개인 취향이다!
              optimizer = Adam(1e-4))                           # 0.0001
hist = model.fit(x_train, y_train,
                 epochs = 20, batch_size = 32,
                 validation_split = 0.3, verbose = 1)

print(hist.history.keys())


# 4. 모델 평가
res = model.evaluate(x_test, y_test, batch_size = 32)
print("loss : ", res[0])
print("acc : ", res[1])


# 5. 시각화
plt.figure(figsize = (10, 6))
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.title('loss')
plt.grid()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['accuracy'], marker = '.', c = 'violet', label = 'acc')
plt.plot(hist.history['val_accuracy'], marker = '.', c = 'green', label = 'val_acc')
plt.title('accuracy')
plt.grid()
plt.ylim(0, 1.0)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc = 'lower right')
plt.show()


'''
Result
loss : 2.4561478313446043
acc  : 0.38269999623298645
'''


'''
과적합 방지
1. 훈련 데이터를 늘린다
2. 피쳐를 늘린다
3. regularization
'''