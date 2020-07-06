# Reuters 뉴스 데이터

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## data load
(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(
    num_words = 10000, test_split = 0.2)
print(x_train.shape, x_test.shape)          # (8982,) (2246,)
print(y_train.shape, y_test.shape)          # (8982,) (2246,)

print(x_train[0])
print(y_train[0])

# print(x_train[0].shape)
print(len(x_train[0]))                      # 87

# 레이블의 범주 확인
category = np.max(y_train) + 1
print(f'카테고리 : {category}')             # 카테고리 : 46

# y의 유일한 원소 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)
print(bbb.shape)            # (46,)

'''주간과제 : pd.groupby()의 사용법 숙지할 것'''

## pad_sequences 사용하여 데이터 shape 맞춰주기
x_train = keras.preprocessing.sequence.pad_sequences(
    x_train, maxlen = 100,
    padding = 'pre', dtype = 'int32')
x_test = keras.preprocessing.sequence.pad_sequences(
    x_test, maxlen = 100,
    padding = 'pre', dtype = 'int32')
print(len(x_train[0]))      # 100
print(len(x_train[-1]))     # 100


## 레이블 원핫인코딩
y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)
print(x_train.shape, x_test.shape)        # (8982, 100) (2246, 100)
print(y_train.shape, y_test.shape)        # (8982, 46) (2246, 46)


## 모델링
model = keras.models.Sequential()
# model.add(keras.layers.Embedding(input_dim = 1000, output_dim = 128,
#                                  input_length = 100))
model.add(keras.layers.Embedding(input_dim = 10000, output_dim = 128))
model.add(keras.layers.LSTM(units = 64))
model.add(keras.layers.Dense(46, activation = keras.activations.softmax))

model.summary()

## 컴파일 및 훈련
model.compile(optimizer = keras.optimizers.Adam(lr = 1e-3),
              loss = keras.losses.categorical_crossentropy,
              metrics = [keras.metrics.categorical_accuracy])
history = model.fit(x_train, y_train,
                    epochs = 10, batch_size = 100,
                    validation_split = 0.2)
acc = model.evaluate(x_test, y_test)[1]
print(f'Accuracy : {acc}')

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker = '.', c = 'red', label = 'TestSet Loss')
plt.plot(y_loss, marker = '.', c = 'blue', label = 'TrainSet Loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()