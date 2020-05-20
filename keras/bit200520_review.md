## 2020.05.20 BIT Review
---

- **LSTM** ; Long Short-Term Memory
**RNN**(순환신경망, Recurrent Neural Network) 기법의 일종
주로 시계열 분석, 자연어 처리 등 순서가 있는 데이터에 적용되는 모델
<br/>

- 공간
  - **스칼라(Scalar)** : 하나의 숫자를 의미함.
  - **벡터(Vector)** : 스칼라가 나열된 형태, 스칼라의 배열
  - **행렬(Matrix)** : 벡터가 나열된 형태, 2차원의 배열
  - **텐서(Tensor)** : 행렬이 나열된 형태, 2차원 이상의 배열
```
ex) [[1, 2, 3], [4, 5, 6]]  ->   (2, 3)의 차원을 갖는 배열
```
<br/>


#### <LSTM 기초 예제 코드 -1>
```
'''LSTM(장단기 메모리; Long Short-Term Memory)'''

# 1. 모듈 임포트
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

# 2. 데이터 구성
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = np.array([4, 5, 6, 7])

print(x.shape)      # res : (4, 3)
print(y.shape)      # res : (4, )
'''y의 차원이 (4, )인 이유: y가 Scala 4개로 이루어진 벡터 한개이기 때문이다'''

# 2-1. 데이터 reshape
x = x.reshape(x.shape[0], x.shape[1], 1)

print(x.shape)


# 3. 모델 구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (3, 1)))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.summary()


# 4. 모델 학습
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x, y, epochs = 100, batch_size = 1)


# 5. 실행
x_input = np.array([5, 6, 7])
# print(x_input.shape)
x_input = x_input.reshape(1, 3, 1)
print(x_input)

y_hat = model.predict(x_input)
print(y_hat)
```


## LSTM의 Parameter 갯수에 대하여.
---

```
# (1)
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (3, 1)))
model.add(Dense(5))
model.add(Dense(1))

# (2)
model.add(LSTM(10, activation = 'relu', input_dim = 1, input_length = 3))
model.add(Dense(5))
model.add(Dense(1))
```

위의 (1)번 코드와 (2)번 코드는 같다.
즉, **input_shape = (3, 1) == [input_dim = 1, input_length = 3]**

- 결과

![model.add(LSTM(10, activation = 'relu', input_shape = (3, 1)))의 결과](https://github.com/seonukim/Study/blob/master/keras/input_shape_res.png)

  ![model.add(LSTM(10, activation = 'relu', input_dim = 1, input_length = 3))](https://github.com/seonukim/Study/blob/master/keras/input_dim_length_res.png)


위에서 보다시피, model.summary()의 결과가 같다고 볼 수 있다.


parameter를 구하는 공식은 아래와 같다.
- ####[(output_nodes + input_dim + 1) * output_nodes] x 4

위의 수식에서,

###### 1. output_nodes = 10
###### 2. input_dim    = 1
###### 3. " + 1 "      = bias 를 나타내며,
###### 4. " x 4 " 를 하는 이유는 아래와 같이 4개의 레이어로 이루어져 있기 때문이다.
####### {1: 'Input Gate', 2: 'Forget Gate', 3: 'Output Gate', 4: 'Gate gate'}
---
여기에서,
- forget gate는 '과거의 정보를 잊기 위한 게이트'
- input gate는 '현재의 정보를 기억하기 위한 게이트'
라고 이해할 수 있다.

![그림](https://github.com/seonukim/Study/blob/master/keras/LSTM_structure.png)

##### 출처 : [RNN과 LSTM을 이해해보자!](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/)
##### 출처 : [StackoverFlow](https://stackoverflow.com/questions/38080035/how-to-calculate-the-number-of-parameters-of-an-lstm-network)





