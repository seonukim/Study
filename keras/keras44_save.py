# keras44_save.py

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
es = EarlyStopping(monitor = 'loss', mode = 'auto', patience = 10)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(10, input_shape = (4, 1)))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


'''
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

'''
model.summary()

# model.save(filepath = ".//model//save_keras44.h5")        # 모델 저장은 .h5 확장자를 쓴다.
# model.save(filepath = "./model/save_keras44.h5")
model.save(filepath = ".\model\save_keras44.h5")
print("저장 잘됐다.")

'''
# 3. 훈련
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x, y, epochs = 1000, batch_size = 1, verbose = 1, callbacks = [es])


# 4. 실행
loss, mse = model.evaluate(x, y)
print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(x, batch_size = 1)
print("=" * 40)
print("y_predict : \n", y_predict)
'''

'''
Result 1)
loss :  0.002710927976295352
mse  :  0.002710927976295352
y_predict :
 [[5.0754447]
  [5.943391 ]
  [6.948891 ]
  [7.9733124]
  [8.987036 ]
  [9.937738 ]]


Result 2)
loss :  8.886436262400821e-05
mse  :  8.886436262400821e-05
y_predict :
 [[ 5.0073833]
  [ 5.9913664]
  [ 7.0102158]
  [ 8.007418 ]
  [ 9.012097 ]
  [10.009922 ]]


Result 3)
loss :  0.015172582119703293
mse  :  0.015172582119703293
y_predict :
 [[5.103075 ]
  [5.9094715]
  [6.9401693]
  [8.013643 ]
  [8.984726 ]
  [9.738817 ]]
'''