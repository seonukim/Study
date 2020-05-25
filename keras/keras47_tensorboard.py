# keras43_dense_split2.py

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
es = EarlyStopping(monitor = 'loss', mode = 'auto', patience = 10)
tb_hist = TensorBoard(log_dir = './graph', histogram_freq = 0,
                      write_graph = True, write_images = True)

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

# 1-2-1. predict 데이터 분할
predict = data[90: , :4]
print(predict)
# predict = predict.reshape(6, 4, 1)
print(predict.shape)


# 1-2-2. train, test 데이터 분할
x = data[:90, :4]
y = data[:90, -1:]
print(x)
print(x.shape)
print(y)
print(y.shape)

# x = x.reshape(90, 4, 1)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True,
    random_state = 1234)

print(x_train.shape)
print(x_test.shape)
print(y_train)
print(y_test)



# 2. 모델 구성
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim = 4))
model.add(Dense(103, activation = 'relu'))
model.add(Dense(150, activation = 'relu'))
model.add(Dense(120, activation = 'relu'))
model.add(Dense(131, activation = 'relu'))
model.add(Dense(113, activation = 'relu'))
model.add(Dense(90, activation = 'relu'))
model.add(Dense(88, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(1))

model.summary()


# 3. 실행 및 훈련
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1000,
          batch_size = 10, verbose = 1,
          callbacks = [es, tb_hist], validation_split = 0.25,
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
loss :  0.16902431845664978
mse  :  0.16902431845664978
y_predict :
 [[94.38897 ]
  [95.38563 ]
  [96.38222 ]
  [97.378845]
  [98.37548 ]
  [99.372116]]


Result 2)
loss :  0.0013877046294510365
mse  :  0.0013877046294510365
y_predict :
 [[94.9642  ]
  [95.96612 ]
  [96.96803 ]
  [97.969955]
  [98.97188 ]
  [99.97378 ]]
'''