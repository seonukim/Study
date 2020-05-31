import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

es = EarlyStopping(monitor = 'loss',
                   mode = 'min',
                   patience = 10)
ss = StandardScaler()
mms = MinMaxScaler()
rs = RobustScaler()
mas = MaxAbsScaler()
pca = PCA(n_components = 6)


''' 1. load data '''
x, y = load_diabetes(return_X_y = True)
print(x.shape)              # (442, 10)
print(y.shape)              # (442,)

# 1-1. preprocessing
pca.fit(x)
x = pca.transform(x)

# 1-2. Scaling
x = mms.fit_transform(x)

# 1-3. data split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)
print(x_train.shape)                # (353, 5)
print(x_test.shape)                 # (89, 5)
print(y_train.shape)                # (353,)
print(y_test.shape)                 # (89,)


''' 2. Modeling _ DNN '''

# 2-1. Sequential Model
model = Sequential()

model.add(Dense(128, activation = 'relu',
                input_shape = (6, )))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(rate = 0.1))
model.add(Dense(1, activation = 'relu'))

'''
# 2-2. Function Model
input1 = Input(shape = (5, ))
layer1 = Dense(128, activation = 'relu')(input1)
layer2 = Dense(128, activation = 'relu')(layer1)
layer3 = Dropout(rate = 0.2)(layer2)
layer4 = Dense(256, activation = 'relu')(layer3)
layer5 = Dense(256, activation = 'relu')(layer4)
layer6 = Dropout(rate = 0.2)(layer5)

output1 = Dense(512, activation = 'relu')(layer6)
output2 = Dense(512, activation = 'relu')(output1)
output3 = Dropout(rate = 0.2)(output2)
output4 = Dense(1, activation = 'relu')(output3)

model = Model(inputs = input1, outputs = output4)
'''
model.summary()


''' 3. Compile & Fitting '''
model.compile(loss = 'mse',
              metrics = ['mse'],
              optimizer = 'adam')
model.fit(x_train, y_train, callbacks = [es],
          epochs = 1000, batch_size = 2,
          validation_split = 0.05, verbose = 1)


''' 4. Evaluate Model '''
res = model.evaluate(x_test, y_test)
print("Result : ", res)
print("loss : ", res[0])
print("mse : ", res[1])

y_predict = model.predict(x_test)
print("=" * 35)
print("Preidict of 1 ~ 5 : \n", y_predict[:5])


''' 5. Evaluation index '''
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
===============
loss :  4353.663864992977
mse :  4353.6640625
===================================
Preidict of 1 ~ 5 : 
 [[218.86763 ]
 [154.4762  ]
 [120.532005]
 [127.056625]
 [199.47823 ]]
===================================
RMSE :  65.98230010102009
===================================
R2 Score :  0.31015900752854664
'''