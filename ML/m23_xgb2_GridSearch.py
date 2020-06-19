# 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피쳐 수를 줄인다.
# 3. regularization     = Dropout과 결과가 비슷 또는 똑같다

from xgboost import XGBClassifier, XGBRegressor, plot_importance      # plot_importance
from sklearn.datasets import load_boston, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt


## 데이터 가져오기
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)      # (506, 13)
print(y.shape)      # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)

## OHE
# encoder = OneHotEncoder()
# encoder.fit(y_train)
# y_train = encoder.transform(y_train)
# y_test = encoder.transform(y_test)

## reshape
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)

# Tree의 Ensemble == Forest
# Forest의 Upgrade == Boosting
# XGBooster의 장점
# 1. 딥러닝 모델에 비해 속도가 빠르다
# 2. 결측치 제거 기능을 자체적으로 제공함
# 3. 하지만 상황에 따라, 판단에 따라 사람이 처리해야 할 필요가 있음
n_estimators = 900              # 앙상블 모델에서 트리의 갯수
learning_rate = 0.01            # 학습률, default == 0.01, 핵심 파라미터 중 하나, 머신러닝 딥러닝 양쪽에서 모두 사용함
colsample_bytree = 0.8          # 성능 좋은 모델들은 통상적으로 0.6 ~ 0.9 사이, tree의 컬럼 샘플 비율을 얼마나 할지 설정
colsample_bylevel = 0.8         # 

max_depth = 7                   # 개별 tree의 깊이, default == 6
n_jobs = -1

params = [
    {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.3, 0.001, 0.01],
     'max_depth': [4, 5, 6]},
    {'n_estimators': [90, 100, 110], 'learning_rate': [0.1, 0.001, 0.01],
     'max_depth': [4, 5, 6], 'colsample_byree': [0.6, 0.9, 1]},
    {'n_estimators': [90, 110], 'learning_rate': [0.1, 0.01, 0.5],
     'max_depth': [4, 5, 6], 'colsample_byree': [0.6, 0.9, 1],
     'colsample_bylevel': [0.6, 0.7, 0.9]}
]

xgb = XGBClassifier(max_depth = max_depth,
                    learning_rate = learning_rate,
                    n_estimators = n_estimators,
                    n_jobs = n_jobs,
                    colsample_bytree = colsample_bytree)
                    #  colsample_bylevel = colsample_bylevel)

model = GridSearchCV(xgb, param_grid = params, cv = 5)

model.fit(x_train, y_train)
print("=" * 40)
print(model.best_estimator_)
print("=" * 40)
print(model.best_params_)
print("=" * 40)
score = model.score(x_test, y_test)
print("=" * 40)
print('점수 : ', score)