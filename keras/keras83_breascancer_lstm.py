import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
es = EarlyStopping(monitor = 'loss',
                   mode = 'min',
                   patience = 10)
pca = PCA(n_components = 20)
mms = MinMaxScaler()

# 1. data
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)              # (569, 30)
print(y.shape)              # (569,)

# 1-1. pca
x = pca.fit_transform(x)

# 1-2. split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)

# 1-3. scaling & reshape
x_train = mms.fit_transform(x_train)
x_train = x_train.reshape(-1, 20, 1)
x_test = x_test.reshape(-1, 20, 1)


# 2. Modeling _ LSTM
model = Sequential()
model.add(LSTM(16, input_shape = (20, 1),
               activation = 'relu',
               return_sequences = True))
model.add(LSTM(16, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'softmax'))

model.summary()


# 3. compile & fit
model.compile(loss = 'binary_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')
model.fit(x_train, y_train, callbacks = [es],
          epochs = 20, batch_size = 1,
          validation_split = 0.05, verbose = 1)


# 4. evaluate
res = model.evaluate(x_test, y_test)
print("Result : ", res)
print("loss : ", res[0])
print("ACc : ", res[1])

pred = model.predict(x_test)
pred = pred.reshape(-1, ).astype('int64')
print("Predict : ", pred[:5])
print("Test data : ", y_test[:5])


'''
loss :  5.245581672902693
ACc :  0.6578947305679321
Predict :  [1 1 1 1 1]
'''