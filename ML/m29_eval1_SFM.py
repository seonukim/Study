'''
m28_eval2와 3을 만들 것

1. eval에 'loss'와 '다른 지표 1개 더 추가
2. earlyStopping 적용
3. plot으로 그릴 것

SelectFromModel에
 1) 회귀         m29_eval1
 2) 이진분류     m29_eval2
 3) 다중분류     m29_eval3


4. 결과는 주석으로 소스 하단에 표시
5. m27 ~ 29까지 완벽 이해할 것!
'''

# 1. 회귀_boston
import numpy as np
import matplotlib.pyplot as plt
import warnings

from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

## 데이터
x, y = load_boston(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)
print(x_train.shape)        # (404, 13)
print(x_test.shape)         # (102, 13)
print(y_train.shape)        # (404,)
print(y_test.shape)         # (102,)