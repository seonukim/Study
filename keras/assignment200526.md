## BIT 2020.05.26 Assignment
___

#### 과제 1) predict값을 0, 1이 나오도록 유도
```python
# 방법 1)
tmp = []
a = np.round(pred[0][0])
b = np.round(pred[1][0])
c = np.round(pred[2][0])
print(a)
print(b)
print(c)
tmp.append([a, b, c])
print(tmp)

# 방법 2) : 제어문 이용
for i in range(len(pred)):
    if i >= 0.5:
        pred[i] = 1
    else:
        pred[i] = 0
print(pred)
```
___

#### 과제 2) y값 data (10, 6) -> (10, 5)로 만들고 predict 구하기
```python
# 1. (10, 6) 차원의 y 데이터를 (10, 5)로 만들기
# 방법 1 : 슬라이싱을 이용하기
from keras.utils import to_categorical
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
y = to_categorical(y)
y = y[:, 1:]

# 방법 2 : Scikit_learn의 OneHotEncoder 클래스 이용하기
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]).reshape(-1, 1)
ohe.fit(y)
y = ohe.transform(y).toarray()

# 2. predict 값 구하기
x_pred = np.array([1, 2, 3, 4, 5])
y_pred = model.predict(x_pred)

# 디코딩
y_pred = np.argmax(y_pred, axis = 1).reshape(-1, )
print("y_pred : \n", y_pred + 1)
'''
Result
loss :  1.4802416026592256
acc  :  0.20000000298023224
y_pred :
 [1 1 1 1 2]
'''
```
___
#### 과제 3) Conv2D의 parameter 갯수 계산법
```python
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(10, (2, 2), input_shape = (10, 10, 1)))
model.add(Conv2D(7, (3, 3)))
model.add(Conv2D(5, (2, 2), padding = 'same'))
model.add(Conv2D(5, (2, 2)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Flatten())
model.add(Dense(1))

model.summary()
```
![conv2d](https://github.com/seonukim/Study/blob/master/keras/Conv2D.png)


위와 같은 model.summary() 결과가 출력되는데,
Conv2D 파라미터의 계산법은 아래와 같다.

###### (필터 크기) x 입력 채널 x 아웃풋 노트 + bias
위의 예에서는, 첫번째 레이어의 파라미터가 50인데 이것은

(2 x 2) x 1 x 10 + 10 == 50
위와 같은 결과에 의해서 나온 것이다.
여기서 각각 (2 x 2) = kernel_size
	  	   1 = input_shape()의 마지막 인자인 channels
      	   10 = 아웃풋
      	   10 = bias 를 의미한다