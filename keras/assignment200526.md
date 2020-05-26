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
