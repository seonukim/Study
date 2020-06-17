import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import LinearSVR, SVR
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
pipe = make_pipeline(StandardScaler(), LinearSVR())
params = {
    'linearsvr__C': [1.0]
}

model = RandomizedSearchCV(pipe, param_distributions = params,
                           cv = 5, n_jobs = -1)

model.fit(x_train, y_train)
print("Score : ", model.best_score_)                # Score :  0.9999964756178837
print("Best Parameters : ", model.best_params_)
'''
Best Parameters :  {'linearsvr__C': 1.0}
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
총지수            103.25              103.30
        쌀         99.53               99.78
        현미       99.62              102.37
        찹쌀       99.63              102.40
        보리쌀    103.65              103.65
        콩        121.00              123.75
        땅콩       95.73              101.22
        혼식곡     89.96               95.21
        밀가루     91.48               99.00
        국수       85.68               95.72
'''

# 너무 좋은데? 과적합의 가능성?