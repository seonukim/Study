# keras42_lstm_split2.py

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
es = EarlyStopping(monitor = 'loss', mode = 'auto', patience = 10)

# 1. 데이터
a = np.array(range(1, 101))
size = 5            # timesteps = 4
print("=" * 40)
print(a.shape)

# LSTM 모델을 완성하시오

# 1-1. 데이터 분할
# 1-1-1. split_x 함수 정의
def split_x(seq, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i : (i + size)]
        # aaa.append([item for item in subset])
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

# 1-2. 데이터 a를 분할 후 x와 y로 분할하기
# 실습1. train, test로 분리할 것 (8:2)
# 실습2. 마지막 6행을 predict로 만들고 싶다
# 실습3. validation을 넣을 것 (train의 20%)
data = split_x(a, size)
print(data)

# 1-2-1. predict 데이터 분할
predict = data[90: , :4]
print(predict)

predict = predict.reshape(6, 4, 1)
print(predict.shape)

# 1-2-2. train, test 데이터 분할
x = data[:90, :4]
y = data[:90, -1:]
print(x)
print(x.shape)
print(y)
print(y.shape)

x = x.reshape(90, 4, 1)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True,
    random_state = 1234)

print(x_train.shape)
print(x_test.shape)
print(y_train)
print(y_test)


# 2. 모델 구성
input1 = Input(shape = (4, 1))
dense1 = LSTM(10, activation = 'relu', return_sequences = True)(input1)
dense2 = LSTM(10, activation = 'relu', return_sequences = True)(dense1)
dense3 = LSTM(8, activation = 'relu')(dense2)
dense4 = Dense(8, activation = 'relu')(dense3)
dense5 = Dense(7, activation = 'relu')(dense4)
dense6 = Dense(9, activation = 'relu')(dense5)

output1 = Dense(8)(dense6)
output2 = Dense(9)(output1)
output3 = Dense(10)(output2)
output4 = Dense(15)(output3)
output5 = Dense(3)(output4)
output6 = Dense(2)(output5)
output7 = Dense(1)(output6)

model = Model(inputs = input1, outputs = output7)

model.summary()


# 3. 실행 및 훈련
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1000,
          batch_size = 1, verbose = 1,
          callbacks = [es], validation_split = 0.25,
          shuffle = True)


# 4. 평가 및 예측
loss, mse = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(predict, batch_size = 1)
print("=" * 40)
print("y_predict : \n", y_predict)
print(y_predict.shape)


'''
Result 1)
loss :  0.2082972996764713
mse  :  0.20829731225967407


Result 2)
loss :  1.17920982837677
mse  :  1.17920982837677


Result 3)
loss :  0.008472939021885395
mse  :  0.008472939021885395
y_predict :
 [[94.94735 ]
  [95.95467 ]
  [96.96222 ]
  [97.96997 ]
  [98.977905]
  [99.98599 ]]
'''