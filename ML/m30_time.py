## 피쳐 엔지니어링

import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

## 데이터
x, y = load_boston(return_X_y = True)
print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)

## 모델링
model = XGBRegressor()
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("Score : ", score)

thresholds = np.sort(model.feature_importances_)
print(thresholds)

import time
start = time.time()
print(start)
for thresh in thresholds:               # 컬럼 수만큼 돈다, 빙글빙글
    selection = SelectFromModel(model, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)

    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 : ", score)

    print("Thresh = %.3f, n = %d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
          score * 100.0))

end = time.time() - start
print("그냥 걸린 시간 : ", end)

import time
start2 = time.time()

for thresh in thresholds:               # 컬럼 수만큼 돈다, 빙글빙글
    selection = SelectFromModel(model, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs = -1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 : ", score)

    print("Thresh = %.3f, n = %d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
          score * 100.0))

end2 = time.time() - start2
print("n_jobs 총 걸린 시간 : ", end2)
'''
Thresh = 0.001, n = 13, R2: 92.21%
Thresh = 0.004, n = 12, R2: 92.16%
Thresh = 0.012, n = 11, R2: 92.03%
Thresh = 0.012, n = 10, R2: 92.19%
Thresh = 0.014, n = 9, R2: 93.08%
Thresh = 0.015, n = 8, R2: 92.37%
Thresh = 0.018, n = 7, R2: 91.48%
Thresh = 0.030, n = 6, R2: 92.71%
Thresh = 0.042, n = 5, R2: 91.74%
Thresh = 0.052, n = 4, R2: 92.11%
Thresh = 0.069, n = 3, R2: 92.52%
Thresh = 0.301, n = 2, R2: 69.41%
Thresh = 0.428, n = 1, R2: 44.98%
'''

# 그리드 서치까지 엮기
# 데이콘 적용, 생체 광학 데이터 -> 소스코드 메일로 제출
# 메일 제목 : 말똥이, 10등