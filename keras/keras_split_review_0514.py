# 1. 모듈 임포트
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# 2. 데이터 구성
# range() 함수는 지정한 숫자 범위 내에서 정수 리스트를 만듦
x = np.array(range(1, 101))
y = np.array(range(1, 101))

# 3. 데이터 분리하기
# 함수를 사용하지 않고 직접 분리
x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]
y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

# 4. 모델 구성
train_test_split()