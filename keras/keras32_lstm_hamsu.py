
# 함수형 모델로 리뉴얼하시오.

# 1. 모듈 임포트
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)

# 2. 데이터 구성
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
              [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
              [9, 10, 11], [10, 11, 12],
              [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x_predict = np.array([50, 60, 70])

'''
# print("x.shape : ", x.shape)        # res : (13, 3)
# print("y.shape : ", y.shape)        # res : (13, )
print(x.shape)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

x_predict = x_predict.reshape(1, 3, 1)

# 3. 모델 구성
# 3-1. 인풋 레이어
input1 = Input(shape = (3, 1))
dense0 = LSTM(10)(input1)
dense1 = Dense(5, activation = 'relu')(dense0)
dense2 = Dense(3, activation = 'relu')(dense1)
dense3 = Dense(3, activation = 'relu')(dense2)
dense4 = Dense(2, activation = 'relu')(dense3)
dense5 = Dense(2, activation = 'relu')(dense4)

# 3-2. 아웃풋 레이어
output1 = Dense(2)(dense5)
output2 = Dense(1)(output1)

# 3-3. 모델링
model = Model(inputs = input1,
              outputs = output2)

model.summary()


# 4. 실행
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x, y,
          epochs = 10000, batch_size = 1,
          callbacks = [early])

x_predict = x_predict.reshape(1, 3, 1)


# 4. 평가 및 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)


'''
'''
from numpy import array
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from sklearn.model_selection import train_test_split
#1. 데이터
x = array([ [1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]
])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) # (4,)
# y2 = array([[4,5,6,7]])     #(1 , 4)
# y3 = array([[4], [5], [6], [7]])   #(4,1)
print("x.shape", x.shape)   #(4 , 3)
print("y.shape", y.shape)   #(4 , ) 스칼라가 4개 input_dim = 1 => 1차원
# print("y2.shape", y2.shape) #(1 , 4) 칼럼이 4개
# print("y3.shape", y3.shape) #(4 , 1) 칼럼이 1개


x = x.reshape(x.shape[0], x.shape[1],1)

# x_train , x_test, y_train, y_test = train_test_split(x,y, train_size = 1, random_state = 66, shuffle = True)
# x = x.reshape(4,3,1)

print(x.shape)

#2. 모델 구성
# model = Sequential()
# # model.add(LSTM(10, activation = 'relu', input_shape = (3,2))) # 실질적으로 행은 큰 영향을 안미치기 때문에 무시 몇개의 칼럼을 가지고 몇개씩 작업을 할 것인가
# model.add(LSTM(130, input_length = 3, input_dim = 1))
# model.add(Dense(149, activation= 'sigmoid'))
# model.add(Dense(35))
# model.add(Dense(60))
# model.add(Dense(90))
# model.add(Dense(690))
# model.add(Dense(610))
# model.add(Dense(470))
# model.add(Dense(250))
# model.add(Dense(1))

input1 = Input(shape = (3,1))
dense1 = LSTM(10)(input1)
output1 = Dense(1)(dense1)

model = Model(input = input1, output = output1)

model.summary()

# 실행
from keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor= 'loss', patience=10, mode = 'min')
model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs=1000, batch_size=10, callbacks=[earlystopping])

x_predict = array([50, 60, 70])
x_predict = x_predict.reshape(1,x_predict.shape[0],1)

print(x_predict)

yhat = model.predict(x_predict)
print(yhat)
'''