import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 와인 데이터 읽기
wine = pd.read_csv('./data/csv/winequality-white.csv',
                   sep = ';', header = 0)

y = wine['quality']
x = wine.drop('quality', axis = 1)      # quality 컬럼을 제외한 나머지를 x에 할당
                                        # quality 컬럼을 드랍한다

print(x.shape)                          # (4898, 11)
print(y.shape)                          # (4898,)
print(y.head())

'''
레이블 클래스가 너무 5와 6에 치중되어 있음
따라서 머신 입장에서는 5, 6 둘 중 하나로 예측하면 쉬운 문제이므로
acc가 높게 나올 수가 없음
그래서 우리는 아래와 같이 y 레이블을 그룹화하여 다시 3개의 등급으로
묶어주는 작업을 해줌
'''
'''
# y 레이블 축소
newlist = []
for i in list(y):
    if i <= 4:
        newlist += [0]
    elif i <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print("acc_score : ", accuracy_score(y_test, y_pred))       # 0.9387755102040817
print("acc       : ", acc)                                  # 0.9387755102040817
'''