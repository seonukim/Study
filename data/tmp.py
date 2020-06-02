print("=" * 20)
print(x[0, :], "\n", y[0])
print("=" * 20)
print(x.shape)                  # (504, 5, 5)
print(y.shape)                  # (504, 1)

x1 = x
x2 = x
print("=" * 20)
print(x1.shape)                 # (252, 5, 5)
print(x2.shape)                 # (252, 5, 5)


# 1-2_1. 데이터 나누기(2)
x1_train, x1_test, y_train, y_test = train_test_split(
    x1, y, test_size = 0.2)
x2_train, x2_test = train_test_split(
    x1, test_size = 0.2)
print("=" * 20)
print(x1_train.shape)           # (403, 5, 5)
print(x1_test.shape)            # (101, 5, 5)
print(x2_train.shape)           # (403, 5, 5)
print(x2_test.shape)            # (101, 5, 5)
print(y_train.shape)            # (403, 1)
print(y_test.shape)             # (101, 1)


# 1-2_2. y데이터 reshape
# y_train = y_train.reshape(-1, )
# y_test = y_test.reshape(-1, )
# print("=" * 20)
# print(y_train.shape)
# print(y_test.shape)


## 1-3. 데이터 scaling
# 1-3_1. 3차원 데이터를 2차원 데이터로 reshape
x1_train = x1_train.reshape(-1, x1_train.shape[1] * x1_train.shape[2])
x1_test = x1_test.reshape(-1, x1_test.shape[1] * x1_test.shape[2])
print("=" * 20)
print(x1_train.shape)
print(x1_test.shape)

x2_train = x2_train.reshape(-1, x2_train.shape[1] * x2_train.shape[2])
x2_test = x2_test.reshape(-1, x2_test.shape[1] * x2_test.shape[2])
print("=" * 20)
print(x2_train.shape)
print(x2_test.shape)

# 1-3_2. MinMaxScaler 적용
mms.fit(x1_train)
x1_train_sc = mms.transform(x1_train)
x1_test_sc = mms.transform(x1_test)
print("=" * 20)
print(x1_train_sc[0, :])
print(x1_test_sc[0, :])

mms.fit(x2_train)
x2_train_sc = mms.transform(x2_train)
x2_test_sc = mms.transform(x2_test)
print("=" * 20)
print(x2_train_sc[0, :])
print(x2_test_sc[0, :])

# 1-3_3. Scaler 적용 후 다시 3차원으로 reshape
x1_train_sc = x1_train_sc.reshape(-1, 25)
x1_test_sc = x1_test_sc.reshape(-1, 25)
print("=" * 20)
print(x1_train_sc.shape)
print(x1_test_sc.shape)

x2_train_sc = x2_train_sc.reshape(-1, 25)
x2_test_sc = x2_test_sc.reshape(-1, 25)
print("=" * 20)
print(x2_train_sc.shape)
print(x2_test_sc.shape)



### 2. 함수형 앙상블 모델링
## 2-1. 첫번째 인풋 모델 구성
input1 = Input(shape = (25, ))
layer1_1 = Dense(32, activation = 'relu')(input1)
layer1_2 = Dense(32, activation = 'relu')(layer1_1)
layer1_3 = Dense(16, activation = 'relu')(layer1_2)
layer1_4 = Dense(16)(layer1_3)

## 2-2. 두번째 인풋 모델 구성
input2 = Input(shape = (25, ))
layer2_1 = Dense(32, activation = 'relu')(input2)
layer2_2 = Dense(32, activation = 'relu')(layer2_1)
layer2_3 = Dense(16, activation = 'relu')(layer2_2)
layer2_4 = Dense(16)(layer2_3)

## 2-3. 두 개의 인풋 모델 병합하기
merge = concatenate([layer1_4, layer2_4])
middle1 = Dense(32, activation = 'relu')(merge)
middle2 = Dense(32, activation = 'relu')(middle1)
middle3 = Dense(16, activation = 'relu')(middle2)

## 2-4. 아웃풋 모델 구성
output = Dense(16, activation = 'relu')(middle3)
output2 = Dense(1, activation = 'relu')(output)

## 2-5. 함수형 모델 정의
model = Model(inputs = [input1, input2],
              outputs = output2)

## 2-6. 모델 요약표
model.summary()



### 3. 컴파일 및 실행
model.compile(loss = 'mse', metrics = ['mse'], optimizer = 'adam')
model.fit([x1_train_sc, x2_train_sc], y_train, callbacks = [es, cp],
          epochs = 1000, batch_size = 1, verbose = 1)


### 4. 모델 평가
res = model.evaluate([x1_test_sc, x2_test_sc], y_test)
print("=" * 20)
print("loss : ", res[0])
print("mse : ", res[1])
