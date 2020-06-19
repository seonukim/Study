# 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피쳐 수를 줄인다.
# 3. regularization     = Dropout과 결과가 비슷 또는 똑같다

from xgboost import XGBClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


## 데이터 가져오기
x, y = load_boston(return_X_y = True)
print(x.shape)      # (506, 13)
print(y.shape)      # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 66)

