## 2020.05.14 수업내용 정리

---
**1. 데이터 분할하기** <br/>
- 데이터는 아래와 같이 분할할 수 있다.
 - 머신이 모델을 학습하는데 필요한 데이터 : 훈련용 데이터 ; **train data**
 - 머신이 학습한 데이터를 평가하는 데이터 : 평가용 데이터 ; **test data**
 - 머신이 학습한 데이터를 검증하는 데이터 : 검증용 데이터 ; **validation data**
   - 검증용 데이터는 훈련용 데이터의 일부를 사용하는 것이 일반적이다.
<br/>

- 데이터를 분할하는 방법
 - 1. 직접 분할하기
 - 2. **train_test_split()** 함수를 이용하여 분할하기
<br/>


#### <1. 예시코드> - 리스트 인덱싱을 이용한 직접 분할
```
# 전체 데이터를 6:2:2로 나눈다.

# 1. 모듈 임포트
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# 2. 데이터 구성
# range() 함수는 지정한 숫자로 범위를 이루는 숫자의 리스트를 만들어준다.
# range(start, stop, step) ; step은 생략 가능, default는 1.
x = np.array(range(1, 101))
y = np.array(range(1, 101))

# 3. 데이터 분할하기
x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

# 4. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 5. 컴파일 및 훈련
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x_train, y_train, batch_size = 1, epochs = 100,
		  validation_data = (x_val, y_val))

# 6. 평가 및 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(x_test)
print("y_predict : ", y_predict)

# 7. RMSE
# RMSE 함수 구현
# 미리 구현된 mean_squared_error()함수에 np.sqrt()(루트)를 씌운다.
def RMSE(y_test, y_predict):
	return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

# 8. R2
print("R2 : ", r2_score(y_test, y_predict))
```
<br/>

#### <2. 예시코드> - train_test_split() 함수를 이용한 데이터 분할
```
# 전체 데이터를 6:2:2로 나눈다.

# 1. 모듈 임포트
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.models_selection import train_test_split
import numpy as np

# 2. 데이터 구성
# range() 함수는 지정한 숫자로 범위를 이루는 숫자의 리스트를 만들어준다.
# range(start, stop, step) ; step은 생략 가능, default는 1.
x = np.array(range(1, 101))
y = np.array(range(1, 101))

# 3. 데이터 분할하기
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = False, test_size = 0.2)
# 검증용 데이터를 만들어 주기 위해 위 함수를 한번 더 적용한다.
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, shuffle = False, test_size = 0.5)

# 4. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 5. 컴파일 및 훈련
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x_train, y_train, batch_size = 1, epochs = 100,
		  validation_data = (x_val, y_val))

# 6. 평가 및 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(x_test)
print("y_predict : ", y_predict)

# 7. RMSE
# RMSE 함수 구현
# 미리 구현된 mean_squared_error()함수에 np.sqrt()(루트)를 씌운다.
def RMSE(y_test, y_predict):
	return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

# 8. R2
print("R2 : ", r2_score(y_test, y_predict))
```
<br/>

#### <2-1. 예시코드>
```
# 위의 코드에서는 검증용 데이터를 분할하기 위해
# train_test_split() 함수를 두 번 사용했는데,
# 아래 코드는 다른 방법으로 검증용 데이터를 사용한다.

# 1. 모듈 임포트
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.models_selection import train_test_split
import numpy as np

# 2. 데이터 구성
# range() 함수는 지정한 숫자로 범위를 이루는 숫자의 리스트를 만들어준다.
# range(start, stop, step) ; step은 생략 가능, default는 1.
x = np.array(range(1, 101))
y = np.array(range(1, 101))

# 3. 데이터 분할하기
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = False, test_size = 0.2)

# 4. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 5. 컴파일 및 훈련
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit(x_train, y_train, batch_size = 1, epochs = 100,
		  validation_split = 0.25))

# 6. 평가 및 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(x_test)
print("y_predict : ", y_predict)

# 7. RMSE
# RMSE 함수 구현
# 미리 구현된 mean_squared_error()함수에 np.sqrt()(루트)를 씌운다.
def RMSE(y_test, y_predict):
	return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

# 8. R2
print("R2 : ", r2_score(y_test, y_predict))
```