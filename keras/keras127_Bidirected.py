from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
imdb
감정에 따라 (긍정적/부정적)으로 라벨된 25,000개의 IMDB 영화 리뷰로 구성된 데이터셋
리뷰는 선행처리 // 각 리뷰는 단어 인덱스(정수)로 구성된 sequence로 인코딩
단어는 데이터내 전체적 사용빈도에 따라 인덱스화
예를 들어, 정수 "3"은 데이터 내에서 세 번째로 빈번하게 사용된 단어를 나타냄
"0"은 특정 단어를 나타내는 것이 아니라 미확인 단어를 통칭
'''

# 1. data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 2000)
# 가장 빈도수가 많은 것부터 2000번쨰까지
# y가 0과 1로 이루어져 있음

print(x_train.shape, x_test.shape)  # (25000,) (25000,)
print(y_train.shape, y_test.shape)  # (25000,) (25000,)

# print(x_train[0])
# print(y_train[0])

# print(x_train[0].shape)   에러 : list는 shape구할 수 없다
print(len(x_train[0]))                  # 218 (이것들의 크기는 일정하지 않다)

# y의 카테고리 계수 출력
category = np.max(y_train) + 1
print('카테고리 :', category)            # 2

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)                          # [0 1]

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)                              # 0 : 12500 , 1 : 12500
print(bbb.shape)                        # (2,)


from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen = 111, padding = 'pre')  # maxlen(최대가 100) / x[0] 문자가 87개, 13개를 0으로 채울 것
x_test = pad_sequences(x_test, maxlen = 111, padding = 'pre')    # truncating(값을 앞 or 뒤에서 잘라서 날리는 것)

print(len(x_train[0]))
print(len(x_train[1]))

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)  # (25000, 111) (25000, 111)


# 2.모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, Dropout, Activation

model = Sequential()
model.add(Embedding(1000, 128, input_length = 111))
# model.add(Embedding(10000, 128))
model.add(LSTM(64))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()
'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2)

acc = model.evaluate(x_test, y_test)[1]
print('\nacc : %.4f' % acc)
# acc : 0.8352 // loss: 0.1733 - acc: 0.9314 - val_loss: 0.5241 - val_acc: 0.8264 <num_word : 2000>
# acc : 0.8312 // loss: 0.2654 - acc: 0.8870 - val_loss: 0.4019 - val_acc: 0.8290 <num_word : 1000>

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c='red', label='TestSet Loss')
plt.plot(y_loss, marker='.', c='blue', label='TrainSet Loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()'''