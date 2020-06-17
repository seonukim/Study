import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
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
pipe = make_pipeline(StandardScaler(), DecisionTreeRegressor())
params = {
    'decisiontreeregressor__criterion': ['mse', 'friedman_mse', 'mae'],
    'decisiontreeregressor__max_depth': [2, 4, 6, 8, 10],
    'decisiontreeregressor__max_features': ['auto', 'sqrt', 'log2']
}

model = RandomizedSearchCV(pipe, param_distributions = params,
                           cv = 5, n_jobs = -1)

model.fit(x_train, y_train)
print("Score : ", model.best_score_)                # Score :  0.9022677116003971
print("Best Parameters : ", model.best_params_)
'''
Best Parameters :  {'decisiontreeregressor__max_features': 'auto',
                    'decisiontreeregressor__max_depth': 10,
                    'decisiontreeregressor__criterion': 'friedman_mse'}
'''

## 예측하기
y_pred = model.predict(test)
print(y_pred)
print(type(y_pred))

pred = pd.DataFrame(y_pred,
                    columns = ['2021'])
print(train['2021'].head(n = 10))
print(pred.head(n = 10))

'''                                     2021
총지수            103.25              104.20
        쌀         99.53               99.34
        현미       99.62               99.34
        찹쌀       99.63               99.34
        보리쌀    103.65              104.20
        콩        121.00              121.03
        땅콩       95.73              102.17
        혼식곡     89.96               94.93
        밀가루     91.48               99.06
        국수       85.68               94.49
'''