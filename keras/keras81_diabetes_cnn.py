import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = 10)
rs = RobustScaler()
pca = PCA(n_components = 2)


# 1. load data
x, y = load_diabetes(return_X_y = True)

# 1-1. PCA
x = pca.fit_transform(x)

# 1-2. scaling
x = rs.fit_transform(x)

# 1-3. split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = 0.2, random_state = 77)

print(x_train.shape)            # (353, 6)
print(y_train.shape)            # (353,)

# 1-4. reshape
x_train = x_train.reshape(x_train.shape[0], 2, 1, 1)
x_test = x_test.reshape(x_test.shape[0], 2, 1, 1)
print(x_train.shape)
print(x_test.shape)


# 2. Modeling _ CNN
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu',
                 input_shape = (2, 1, 1)))
model.add(Conv2D(filters = 16, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Dropout(rate = 0.15))

model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = 'relu'))
model.add(Dropout(rate = 0.2))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

model.summary()


# 3. compile & fitting
model.compile(loss = 'mse',
              metrics = ['mse'],
              optimizer = 'adam')
hist = model.fit(x_train, y_train, callbacks = [es],
                 epochs = 1000, batch_size = 2,
                 validation_split = 0.2, verbose = 1)

print(hist.history.keys())


# 4. Evaluating model
res = model.evaluate(x_test, y_test)
print("Result : ", res)
print("loss : ", res[0])
print("mse : ", res[1])

y_predict = model.predict(x_test)
print("=" * 35)
print("Predict of 1 ~ 5 : \n", y_predict[:5])


# 5. Evaluation index
# 5-1. RMSE
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("=" * 35)
print("RMSE : ", RMSE(y_test, y_predict))

# 5-2. R2 Score
print("=" * 35)
print("R2 Score : ", r2_score(y_test, y_predict))


'''
Result
loss :  3965.1302065045647
mse :  3965.13037109375
===================================
Predict of 1 ~ 5 : 
 [[138.67142]
 [111.63252]
 [176.2782 ]
 [114.03755]
 [216.24779]]
===================================
RMSE :  62.969278327593734
===================================
R2 Score :  0.37172246879497883
'''