# mae 는
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Dropout, MaxPool2D, Flatten
from sklearn.metrics import mean_absolute_error
# import xgboost
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col=0)

train = train.interpolate()
test = test.interpolate()

train = train.fillna(train.mean())
test = test.fillna(train.mean())

print(train.shape)
print(test.shape)

train = np.array(train)
x_predict = np.array(test)

x = train[:,:71]
y = train[:,71:]
print(x.shape)      # (10000, 71)
print(y.shape)      # (10000, 4)

# 전처리
from sklearn.preprocessing import RobustScaler, StandardScaler
scaler = RobustScaler()
scaler.fit(x) 
x = scaler.transform(x)
x_predict = scaler.transform(x_predict)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=20)
# pca.fit(x)
# x = pca.transform(x) 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=43)

# model = Sequential()
# model.add(Dense(52, input_dim=(71))) 
# model.add(Dense(228))
# model.add(Dense(356))
# model.add(Dropout(0.2))
# model.add(Dense(250))
# model.add(Dense(164))
# model.add(Dropout(0.4))
# model.add(Dense(64))
# model.add(Dense(4, activation='relu')) 
# model.summary()
# model.compile(loss='mae', optimizer='adam', metrics=['mae']) 
# model.fit(x_train,y_train, epochs=500, batch_size=50)
# y_pred = model.predict(x_test)
# mse = mean_absolute_error(y_pred, y_test)
# print(mse)

model = RandomForestRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_test) # 회귀던 분류던 사용할 수 있음
mae = mean_absolute_error(y_pred, y_test)
print('mae 는', mae)
