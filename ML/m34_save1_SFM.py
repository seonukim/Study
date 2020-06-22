'''
m29_eval1_SFM.py
m29_eval2_SFM.py
m29_eval3_SFM.py에 save를 적용하시오

save 이름에는 평가지표를 첨가해서
가장 좋은 SFM용 save파일을 나오도록 할것
'''

# 200622_25
# boston / model save


import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score


### 데이터 ###
x, y = load_boston(return_X_y=True)
print(x.shape)      # (506, 13)
print(y.shape)      # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)


### 기본 모델 ###
model = XGBRegressor(n_estimators=300, learning_rate=0.1, n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 :', score)

#== Default R2 : 0.9313126937746082 ==#


### feature engineering ###
thresholds = np.sort(model.feature_importances_)
print(thresholds)


models = []
res = np.array([])

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    model2 = XGBRegressor(n_estimators=300, learning_rate=0.1, n_jobs=-1)
    model2.fit(select_x_train, y_train, verbose=False, eval_metric=['logloss','rmse'],
                eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                early_stopping_rounds=20)
    
    y_pred = model2.predict(select_x_test)
    score = r2_score(y_test, y_pred)
    shape = select_x_train.shape
    models.append(model2)           # 모델을 배열에 저장
    print(thresh, score)
    res = np.append(res, score)     # 결과값을 전부 배열에 저장
print(res.shape)
best_idx = res.argmax()             # 결과값 최대값의 index 저장
score = res[best_idx]               # 위 인덱스 기반으로 점수 호출
total_col = x_train.shape[1] - best_idx # 전체 컬럼 계산
models[best_idx].save_model(f'./model/xgb_save/boston--{score}--{total_col}--.model')    # 인덱스 기반으로 모델 저장

# Thresh=0.001, n=13, R2: 93.29%
# Thresh=0.002, n=12, R2: 93.25%
# Thresh=0.010, n=11, R2: 93.27%
# Thresh=0.010, n=10, R2: 93.40%
# Thresh=0.013, n=9, R2: 93.20%
# Thresh=0.014, n=8, R2: 93.49%
# Thresh=0.017, n=7, R2: 93.52%
# Thresh=0.026, n=6, R2: 94.02%
# Thresh=0.036, n=5, R2: 92.82%
# Thresh=0.042, n=4, R2: 92.00%
# Thresh=0.045, n=3, R2: 89.06%
# Thresh=0.246, n=2, R2: 81.25%
# Thresh=0.537, n=1, R2: 68.26%