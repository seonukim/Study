# 모듈 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, Input, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# ModelCheckpoint 경로 변수 생성
modelpath = './data/Check-{epoch:02d} - {val_loss:.4f}.hdf5'

# 객체 생성
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 100)
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     mode = 'auto', save_best_only = True,
                     save_weights_only = False)
ss = StandardScaler()
mms = MinMaxScaler()
mas = MaxAbsScaler()
rs = RobustScaler()



### 1. 데이터
# 주가는 stock price; 축약하여 sp로 표기함
samsung_sp = pd.read_csv('./data/csv/samsung.csv',
                         index_col = 0, header = 0,
                         sep = ',', encoding = 'cp949')
print(samsung_sp.shape)                 # (700, 1)
# print(samsung_sp.tail())            

hite_sp = pd.read_csv('./data/csv/hite.csv',
                      index_col = 0, header = 0,
                      sep = ',', encoding = 'cp949')
print("=" * 40)
print(hite_sp.shape)                    # (720, 5)
# print(hite_sp)


## 1-1. 데이터 전처리
# 1-1_1. 결측치 제거하기  - dropna() 메서드를 사용함
samsung_sp = samsung_sp.dropna()
print("=" * 40)
print(samsung_sp.shape)                 # (509, 1)
print(samsung_sp.tail())

hite_sp.iloc[0, 1:] = hite_sp.iloc[0, 1:].fillna(0)       # 첫 행의  NaN을 채워줌
hite_sp = hite_sp.dropna()
print("=" * 40)
print(hite_sp.shape)                    # (509, 5)
print(hite_sp.head())

print("=" * 40)
print(samsung_sp.isna())                # 확인함
print(hite_sp.isna())                   # 확인함


# 1-1_2. string 타입 -> int 타입으로 변경해주기
# comma 제거 함수 정의
def remove_comma(x):
    return x.replace(',', '')

# 1-1_2_1. 삼성전자 주가 데이터 "," 제거
samsung_sp['시가'] = samsung_sp['시가'].apply(remove_comma)
print("=" * 40)
print(samsung_sp)
print(samsung_sp.dtypes)                # object

# 1-1_2_2. 하이트진로 주가 데이터 "," 제거
hite_sp['시가'] = hite_sp['시가'].astype(str)
hite_sp['고가'] = hite_sp['고가'].astype(str)
hite_sp['저가'] = hite_sp['저가'].astype(str)
hite_sp['종가'] = hite_sp['종가'].astype(str)
hite_sp['거래량'] = hite_sp['거래량'].astype(str)

hite_sp['시가'] = hite_sp['시가'].apply(remove_comma)
hite_sp['고가'] = hite_sp['고가'].apply(remove_comma)
hite_sp['저가'] = hite_sp['저가'].apply(remove_comma)
hite_sp['종가'] = hite_sp['종가'].apply(remove_comma)
hite_sp['거래량'] = hite_sp['거래량'].apply(remove_comma)
print("=" * 40)
print(hite_sp)
print(hite_sp.dtypes)                   # object

# 1-1_2_3. 정수형으로 데이터 변환
samsung_sp['시가'] = samsung_sp['시가'].astype('int64')
print("=" * 40)
print(samsung_sp.dtypes)                # int64

hite_sp['시가'] = hite_sp['시가'].astype('int64')
hite_sp['고가'] = hite_sp['고가'].astype('int64')
hite_sp['저가'] = hite_sp['저가'].astype('int64')
hite_sp['종가'] = hite_sp['종가'].astype('int64')
hite_sp['거래량'] = hite_sp['거래량'].astype('int64')
print("=" * 40)
print(hite_sp.dtypes)                   # int64

# 1-1_2_4. 하이트진로 주가 데이터 첫행 0 값 대체하기
hite_sp = hite_sp.replace({'고가': 0}, {'고가': 39500})
hite_sp = hite_sp.replace({'저가': 0}, {'저가': 38500})
hite_sp = hite_sp.replace({'종가': 0}, {'종가': 38800})
hite_sp = hite_sp.replace({'거래량': 0}, {'거래량': 638583})
print("=" * 40)
print(hite_sp.head())


# 1-1_3. 주가 오름차순 정렬하기
samsung_sp = samsung_sp.sort_values(['일자'], ascending = [True])
hite_sp = hite_sp.sort_values(['일자'], ascending = [True])
print("=" * 40)
print(samsung_sp)
print(hite_sp)


# 1-1_4. 하이트진로 주가 데이터 1~4열 슬라이싱('시가' 컬럼만 남기기)
hite_sp = hite_sp.iloc[:, :1]
print(hite_sp)


# 1-1_5. numpy 파일로 저장
samsung_sp = samsung_sp.values
hite_sp = hite_sp.values
print("=" * 40)
print(type(samsung_sp))                 # <class 'numpy.ndarray'>
print(type(hite_sp))                    # <class 'numpy.ndarray'>
print("=" * 40)
print(samsung_sp.shape)                 # (509, 1)
print(hite_sp.shape)                    # (509, 1)

np.save('./data/samsung_sp.npy', arr = samsung_sp)
np.save('./data/hite_sp.npy', arr = hite_sp)


# 1-1_6. numpy 형식 데이터 불러오기
np_samsung_sp = np.load('./data/samsung_sp.npy')
np_hite_sp = np.load('./data/hite_sp.npy')
print("=" * 40)
print(np_samsung_sp)
print(np_hite_sp)
print("=" * 40)
print(np_samsung_sp.shape)              # (509, 1)
print(np_hite_sp.shape)                 # (509, 1)


## 1-2. 데이터 나누기
def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy(np_hite_sp, 2, 1)
x2, y2 = split_xy(np_samsung_sp, 2, 1)
print("=" * 40)
print(x1[0, :], "\n", y1[0])
print(x1.shape)                 # (507, 2, 1)
print(y1.shape)                 # (507, 1)
print(x2.shape)                 # (507, 2, 1)
print(y2.shape)                 # (507, 1)


# 1-2_1. 데이터 나누기(2)
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, test_size = 0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, test_size = 0.3)
print("=" * 40)
print(x1_train.shape)                   # (354, 2, 1)
print(x1_test.shape)                    # (153, 2, 1)
print(y1_train.shape)                   # (354, 1)
print(y1_test.shape)                    # (153, 1)
print("=" * 40)
print(x2_train.shape)                   # (354, 2, 1)
print(x2_test.shape)                    # (153, 2, 1)
print(y2_train.shape)                   # (354, 1)
print(y2_test.shape)                    # (153, 1)


## 1-3. 데이터 scaling
# 1-3_1. 3차원 데이터를 2차원 데이터로 reshape
x1_train = x1_train.reshape(-1, x1_train.shape[1] * x1_train.shape[2])
x1_test = x1_test.reshape(-1, x1_test.shape[1] * x1_test.shape[2])
print("=" * 40)
print(x1_train.shape)                   # (354, 2)
print(x1_test.shape)                    # (153, 2)

x2_train = x2_train.reshape(-1, x2_train.shape[1] * x2_train.shape[2])
x2_test = x2_test.reshape(-1, x2_test.shape[1] * x2_test.shape[2])
print("=" * 40)
print(x2_train.shape)                   # (354, 2)
print(x2_test.shape)                    # (153, 2)


# 1-3_2. MinMaxScaler 적용
mms.fit(x1_train)
x1_train_sc = mms.transform(x1_train)
x1_test_sc = mms.transform(x1_test)
print("=" * 40)
print(x1_train_sc[0, :])
print(x1_test_sc[0, :])

mms.fit(x2_train)
x2_train_sc = mms.transform(x2_train)
x2_test_sc = mms.transform(x2_test)
print("=" * 40)
print(x2_train_sc[0, :])
print(x2_test_sc[0, :])


'''
### 2. 함수형 앙상블 모델링
## 2-1. 첫번째 인풋 모델 구성
input1 = Input(shape = (2, ))
layer1_1 = Dense(32, activation = 'relu')(input1)
layer1_2 = Dense(16)(layer1_1)

## 2-2. 두번째 인풋 모델 구성
input2 = Input(shape = (2, ))
layer2_1 = Dense(16, activation = 'relu')(input2)
layer2_2 = Dense(16)(layer2_1)

## 2-3. 두 개의 인풋 모델 병합하기
merge = concatenate([layer1_2, layer2_2])
middle1 = Dense(8, activation = 'relu')(merge)

## 2-4. 아웃풋 모델 구성
output = Dense(8, activation = 'relu')(merge)
output2 = Dense(1, activation = 'relu')(output)

## 2-5. 함수형 모델 정의
model = Model(inputs = [input1, input2],
              outputs = output2)

## 2-6. 모델 요약표
model.summary()



### 3. 컴파일 및 실행
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'rmsprop')
model.fit([x1_train_sc, x2_train_sc], y2_train, callbacks = [cp],
          validation_split = 0.05, epochs = 1000, batch_size = 2, verbose = 1)
'''

model = load_model('./data/Check-975 - 648347.6816.hdf5')



### 4. 모델 평가
res = model.evaluate([x1_test_sc, x2_test_sc], y2_test, batch_size = 1)
print("=" * 40)
print("loss : ", res[0])
print("mse : ", res[1])

y_predict = model.predict([x1_test_sc, x2_test_sc])
# y_predict = np.argmax(y_predict, axis = 1)
print("=" * 40)
for i in range(3):
    print('종가 : ', y2_test[i], '/ 예측가 : ', y_predict[i])



### 5. 평가 지표
## 5-1. RMSE 지표
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("=" * 40)
print("RMSE : ", RMSE(y2_test, y_predict))

## 5-2. R2 Score 지표
print("=" * 40)
print("R2 Score : ", r2_score(y2_test, y_predict))
