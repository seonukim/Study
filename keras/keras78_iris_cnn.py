import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Input
from keras.layers import Dropout, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

es = EarlyStopping(monitor = 'loss',
                   mode = 'min',
                   patience = 10)
ss = StandardScaler()
mms = MinMaxScaler()
rs = RobustScaler()
mas = MaxAbsScaler()
pca = PCA(n_components = 2)


''' 1. load data '''
x, y = load_iris(return_X_y = True)
print(x.shape)                  # (150, 4)
print(y.shape)                  # (150,)

# 1-1. preprocessing
# pca.fit(x)
# x = pca.transform(x)

# 1-2. Scaling
x = mas.fit_transform(x)

# 1-3. data split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)
print(x_train.shape)            # (120, 2)
print(x_test.shape)             # (30, 2)
print(y_train.shape)            # (120,)
print(y_test.shape)             # (30,)

# 1-4. data reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)
print(x_train.shape)
print(x_test.shape)

# 1-5. one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train[0])
print(y_test[0])


''' 2. Modeling CNN '''
# 2-1. Sequential Model
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                 input_shape = (4, 1, 1), padding = 'same',
                 activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 128, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Dropout(rate = 0.2))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

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
loss :  0.06103597581386566
acc :  0.9555556178092957
===================================
Predict of 1 ~ 5 : 
 [1 1 2 1 0]


 Result 2)
 - PCA = 2, Scaler 미적용
loss :  0.20910383760929108
acc :  0.9555556178092957
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]


 Result 3)
 - PCA = 2, StadardScaler 적용
loss :  0.2907279431819916
acc :  0.9555556178092957
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]


 Result 4)
 - PCA = 2, MinMaxScaler 적용
loss :  0.0866202637553215
acc :  0.9555556178092957
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]


 Result 5)
 - PCA = 2, RobustScaler 적용
loss :  0.2585287094116211
acc :  0.9555556178092957
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]


 Result 6)
 - PCA = 2, MaxAbsScaler 적용
loss :  0.23430992662906647
acc :  0.9555556178092957
===================================
Predict of 1 ~ 5 : 
 [1 2 2 1 0]


 Result 7)
 - PCA 미적용, StandardScaler 적용
loss :  1.1602023839950562
acc :  0.8999999761581421
===================================
Predict of 1 ~ 5 : 
 [1 1 2 1 0]


 Result 8)
 - PCA 미적용, MinMaxScaler 적용
loss :  0.6221228837966919
acc :  0.8666666746139526
===================================
Predict of 1 ~ 5 : 
 [1 1 2 1 0]


 Result 9)
 - PCA 미적용, RobustScaler 적용
loss :  4.560882568359375
acc :  0.8666666746139526
===================================
Predict of 1 ~ 5 : 
 [1 1 2 1 0]


 Result 10)
 - PCA 미적용, MaxAbsScaler 적용
loss :  0.14479361474514008
acc :  0.8666666746139526
===================================
Predict of 1 ~ 5 : 
 [1 1 2 1 0]
'''