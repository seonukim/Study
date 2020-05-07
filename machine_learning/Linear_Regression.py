# 1. 데이터 준비 및 확인
import pandas as pd
boston = pd.read_csv('https://raw.githubusercontent.com/'
                     'rasbt/python-machine-learning-book-2nd-edition/'
                     'master/code/ch10/housing.data.txt',
                     header = None,
                     sep = '\s+')
boston.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
                  'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                  'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(boston.head(n = 10))

# 2. 데이터 주요 특징 시각화
# 2-1. 산점행렬(scatterplot matrix) - 변수 간의 상관관계 확인
import matplotlib.pyplot as plt
import seaborn as sns
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(boston[cols], height = 2.5)
plt.tight_layout()
plt.show()

# 2-2. 상관관계 행렬(correlation matrix)
import numpy as np
cm = np.corrcoef(boston[cols].values.T)
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm,
                 cbar = True,
                 annot = True,
                 square = True,
                 fmt = '.2f',
                 annot_kws = {'size': 15},
                 yticklabels = cols,
                 xticklabels = cols)
plt.tight_layout()
plt.show()

# 3. 최소 제곱 선형 회귀 모델 구현
class LinearRegressionGD(object):
    
    def __init__(self, eta = 0.001, n_iter = 20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

# 3-1. 표준화 전처리 및 모델 훈련
X = boston[['RM']].values
y = boston['MEDV'].values
from sklearn.preprocessing import StandardScaler
"""사이킷런의 전처리 패키지에서 표준화 스케일러를 임포트"""
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# 경사하강법 알고리즘이 비용 함수의 최솟값으로 수렴하는지 확인
plt.plot(range(1, lr.n_iter + 1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

# 3-2. 훈련 샘플의 산점도와 회귀 직선 나타내기
# 산점도와 회귀 직선을 그려주는 헬퍼 함수 생성
def lin_regplot(X, y, model):
    plt.scatter(X, y, c = 'steelblue', edgecolor = 'white', s = 70)
    plt.plot(X, model.predict(X), color = 'black', lw = 2)
    return None

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()

# 예측한 가격을 $1,000 단위 가격으로 되돌리기
num_rooms_std = sc_x.transform([[5.0]])
price_std = lr.predict(num_rooms_std)
print("$1,000 단위 가격: %.3f" % \
      sc_y.inverse_transform(price_std))

