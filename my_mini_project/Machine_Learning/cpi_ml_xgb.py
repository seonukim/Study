import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scaler = StandardScaler()

## 데이터
train = pd.read_csv('./my_mini_project/cpi_train.csv',
                    index_col = 0, header = 0,
                    encoding = 'cp949')
test = pd.read_csv('./my_mini_project/cpi_test.csv',
                    index_col = 0, header = 0,
                    encoding = 'cp949')
print(train.shape)          # (461, 47)
print(test.shape)           # (461, 46)

## 결측치 확인
print(train.isnull().sum())
print(test.isnull().sum())

## 선형보간법을 사용하여 먼저 결측치 처리해보기
train = train.interpolate()
test = test.interpolate()
print(train.isnull().sum())
print(test.isnull().sum())


## 데이터 나누기
x = train.iloc[:, :46]
y = train.iloc[:, 46:]
print(x.shape)              # (461, 46)
print(y.shape)              # (461, 1)

## train, test로 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = False)
print(x_train.shape)        # (368, 46)
print(x_test.shape)         # (93, 46)
print(y_train.shape)        # (368, 1)
print(y_test.shape)         # (93, 1)

## 모델링
kf = KFold(n_splits = 5)
xgb = XGBRegressor()
params = {'nthread':[4],
          'objective':['reg:linear'],
          'learning_rate': [.03, 0.05, .07],
          'max_depth': [5, 6, 7],
          'min_child_weight': [4],
          'silent': [1],
          'subsample': [0.7],
          'colsample_bytree': [0.7],
          'n_estimators': [100, 200, 300, 400, 500]}

model = RandomizedSearchCV(xgb, param_distributions = params,
                           cv = 5, n_jobs = -1)

model.fit(x_train, y_train)
print("Score : ", model.best_score_)                # Score :  0.971320430617047
print("Best Parameters : ", model.best_params_)
'''
Best Parameters :  {'subsample': 0.7,
                    'silent': 1,
                    'objective': 'reg:linear',
                    'nthread': 4,
                    'n_estimators': 500,
                    'min_child_weight': 4,
                    'max_depth': 7,
                    'learning_rate': 0.03,
                    'colsample_bytree': 0.7}
'''

## 예측하기
y_pred = model.predict(test)
print(y_pred)
print(type(y_pred))

pred = pd.DataFrame(y_pred,
                    columns = ['2021'])
print(train['2021'].head(n = 10))
print(pred.head(n = 10))

'''                 2021                2021
총지수            103.25              103.15
        쌀         99.53               99.60
        현미       99.62              101.88
        찹쌀       99.63              102.06
        보리쌀    103.65              103.73
        콩        121.00              122.42
        땅콩       95.73              101.47
        혼식곡     89.96               96.02
        밀가루     91.48               99.38
        국수       85.68               97.77
'''

# 괜찮은 결과다, 과적합의 가능성이 있는 linearSVR보다 신뢰도가 높음