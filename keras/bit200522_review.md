## 2020.05.22 BIT Review

#### LSTM 레이어를 여러 개 연결하는 방법
```python
model = Sequential()
model.add(LSTM(10, activation = 'relu', 
			   input_length = 3, input_dim = 1,
               return_sequences = True))
model.add(LSTM(10, return_sequences = True))
```

LSTM은 return_sequences라는 파라미터를 지원한다.
return_sequences를 넣지 않고 LSTM을 연결하면 아래와 같은 에러메시지가 출력된다

++**ValueError: Input 0 is incompatible with layer lstm_2: expectied ndim=3, found ndim=2**++

위의 에러는 "LSTM은 3차원의 배열을 입력해야 하지만, 찾은 건 2차원 배열이다" 라는 의미이다.

아래는 return_sequences 파라미터에 대해 정리한 내용.
- LSTM은 3차원의 배열을 넣어줘야 하는데, (batch_size, timesteps, features)
- 우리가 입력한 차원은 2차원이다. ; Dense(row, column) -> 2차원
- return_sequences는 이전 차원을 그대로 유지해주는 기능을 하는 parameter이다.
- return_sequences의 값은 boolean, True of False

___

#### RNN 모델들의 파라미터 갯수 계산법
- **SimpleRNN** : 4 x (input_dim + bias(1) + output) x output
- **GRU**		: 3 x (input_dim + bias(1) + output) x output
- **LSTM**		: 1 x (input_dim + bias(1) + output) x output


- output이 한번 더 곱해지는 이유는 역전파 현상 때문이다.
___

#### Scikit-learn의 Scaler 사용하기
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

# scaler = MinMaxScaler()       # 최소/최대값이 0, 1이 되도록 스케일링
# scaler = StandardScaler()     # 평균이 0이고 표준편차가 1인 정규분포가 되도록 스케일링
# scaler = MaxAbsScaler()       # 최대절대값과 0이 각각 1, 0이 되도록 스케일링
# scaler = RobustScaler()       # 중위수와 4분위수 사용하여 이상치의 영향을 최소화함

scaler.fit(x)
x = scaler.transform(x) 
```
___

#### split_x 함수 정의하기
```python
def split_x(seq, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i : (i + size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)
```