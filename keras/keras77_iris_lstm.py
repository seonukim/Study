import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import load_iris

es = EarlyStopping(monitor = 'loss',
                   mode = 'min',
                   patience = 10)
pca = PCA(n_components = 2)
ss = StandardScaler()
mms = MinMaxScaler()
mas = MaxAbsScaler()
rs = RobustScaler()


''' 1. load data '''
x, y = load_iris(return_X_y = True)
print(x.shape)                  # (150, 4)
print(y.shape)                  # (150,)

# 1-1. preprocessing
# pca.fit(x)
# x = pca.transform(x)

# 1-2. normalization
x = mas.fit_transform(x)

# 1-3. data split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)

print(x_train.shape)            # (120, 4)
print(x_test.shape)             # (30, 4)
print(y_train.shape)            # (120,)
print(y_test.shape)             # (30,)

# 1-4. One Hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 1-5. data reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print(x_train.shape)
print(x_test.shape)


''' 2. Modeling LSTM '''
# 2-1. Sequential Model
model = Sequential()

model.add(LSTM(128, input_shape = (4, 1),
               activation = 'relu',
               return_sequences = True))
model.add(LSTM(128, activation = 'relu',
               return_sequences = True))
model.add(LSTM(256, activation = 'relu',
               return_sequences = False))
model.add(Dropout(rate = 0.2))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(3, activation = 'softmax'))


'''
# 2-2. Function Model
input1 = Input(shape = (2, 1))
layer1 = LSTM(128, activation = 'relu',
              return_sequences = True)(input1)
layer2 = LSTM(128, activation = 'relu',
              return_sequences = True)(layer1)
layer3 = Dropout(rate = 0.2)(layer2)
layer4 = LSTM(256, activation = 'relu')(layer3)
layer5 = Dense(256, activation = 'relu')(layer4)
layer6 = Dropout(rate = 0.2)(layer5)

output1 = Dense(512, activation = 'relu')(layer6)
output2 = Dense(512, activation = 'relu')(output1)
output3 = Dense(3, activation = 'sigmoid')(output2)
'''

model.summary()


''' 3. Compile & Fitting '''
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
hist = model.fit(x_train, y_train, callbacks = [es],
                 epochs = 1000, batch_size = 1,
                 validation_split = 0.05, verbose = 1)
print(hist.history.keys())


''' 4. Evaluate Model '''
res = model.evaluate(x_test, y_test)
print("Result : ", res)
print("loss : ", res[0])
print("acc : ", res[1])

pred = model.predict(x_test)
pred = np.argmax(pred, axis = 1)
print("=" * 35)
print("Predict of 1 ~ 5 : \n", pred[:5])

y_test = np.argmax(y_test, axis = 1)
print("=" * 35)
print("Test data of 1 ~ 5 : \n", y_test[:5])


'''
Result 1)
- PCA, Scaler 미적용
loss :  0.39339929819107056
acc :  0.9000000357627869
===================================
Predict of 1 ~ 5 : 
 [1 1 1 1 0]


 Result 2)
 - PCA = 2, Scaler 미적용
loss :  0.495330810546875
acc :  0.9333333373069763
===================================
Predict of 1 ~ 5 : 
 [1 2 2 0 0]


 Result 3)
 - PCA = 2, StandardScaler 적용
loss :  0.21440400183200836
acc :  0.9555556178092957
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]


 Result 4)
 - PCA = 2, MinMaxScaler 적용
loss :  0.11141034960746765
acc :  0.9555556178092957
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]


 Result 5)
 - PCA = 2, RobustScaler 적용
loss :  0.20607754588127136
acc :  0.9333333373069763
===================================
Predict of 1 ~ 5 : 
 [1 2 2 0 0]


 Result 6)
 - PCA = 2, MaxAbsScaler 적용
loss :  0.19592995941638947
acc :  0.9333333373069763
===================================
Predict of 1 ~ 5 : 
 [1 1 2 1 0]


 Result 7)
  - PCA 미적용, StandardScaler 적용
loss :  0.38097578287124634
acc :  0.888888955116272
===================================
Predict of 1 ~ 5 : 
 [1 1 2 1 0]


 Result 8)
 - PCA 미적용, MinMaxScaler 적용
loss :  0.1580464392900467
acc :  0.9333333373069763
===================================
Predict of 1 ~ 5 : 
 [1 1 2 1 0]


 Result 9)
 - PCA 미적용, RobustScaler 적용
loss :  0.22718264162540436
acc :  0.8999999761581421
===================================
Predict of 1 ~ 5 : 
 [1 1 2 1 0]


 Result 10)
 - PCA 미적용, MinAbsScaler 적용
loss :  0.32244810461997986
acc :  0.8333333730697632
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]
'''