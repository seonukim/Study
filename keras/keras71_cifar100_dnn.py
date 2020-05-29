import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten
from keras.layers import MaxPooling2D, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.datasets import cifar100
modelfath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'

# 클래스 객체 생성
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
cp = ModelCheckpoint(filepath = modelfath, monitor = 'val_loss',
                     mode = 'auto', save_best_only = True)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape)            # (50000, 32, 32, 3)
print(x_test.shape)             # (10000, 32, 32, 3)
print(y_train.shape)            # (50000, 1)
print(y_test.shape)             # (10000, 1)

# 1-1. 정규화
x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.0
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.0

# 1-2. OHE
y_train = np_utils.to_categorical(y_train, num_classes = 100)
y_test = np_utils.to_categorical(y_test, num_classes = 100)
print(y_train.shape)
print(y_test.shape)


# 2. 모델링

input1 = Input(shape = (32, 32, 3))
layer1 = Dense(16, activation = 'relu')(input1)
layer2 = Dense(16, activation = 'relu')(layer1)
layer3 = Dropout(rate = 0.15)(layer2)
layer4 = Dense(32, activation = 'relu')(layer3)
layer5 = Dense(32, activation = 'relu')(layer4)
layer6 = Dropout(rate = 0.15)(layer5)
layer7 = Flatten()(layer6)

output1 = Dense(64, activation = 'relu')(layer7)
output2 = Dense(64, activation = 'relu')(output1)
output3 = Dense(100, activation = 'softmax')(output2)

model = Model(inputs = input1, outputs = output3)

'''
model = Sequential()

model.add(Dense(16, input_shape = (32, 32, 3), activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(rate = 0.15))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(rate = 0.15))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(rate = 0.15))
model.add(Dense(100, activation = 'softmax'))
'''

model.summary()


# 3. 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')
hist = model.fit(x_train, y_train, callbacks = [es, cp],
                 epochs = 50, batch_size = 1024,
                 validation_split = 0.001, verbose = 1)

print(hist.history.keys())


# 4. 모델 평가
res = model.evaluate(x_test, y_test)
print("Result : ", res)


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
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc = 'lower right')
plt.show()


'''
Result
loss : 3.2346978572845457
acc  : 0.23389999568462372
'''