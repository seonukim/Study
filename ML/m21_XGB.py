from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size  = 0.8, random_state = 42
)

# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()

# max_features : 기본값 써라!
# n_estimators : 클 수록 좋다!, 단점 메모리 짱 차지, 기본 값 100
# n_jobs = -1  : 병렬 처리

model.fit(x_train, y_train)

acc = model.score(x_test , y_test)

print(model.feature_importances_)
print(acc)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align = 'center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()