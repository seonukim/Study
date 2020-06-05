import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./data/csv/boston_house_prices.csv',
                   header = 1, sep = ',', index_col = None)
print(df.head())

print(df.isna())


x = df.iloc[:, :13]
y = df.iloc[:, -1:]
print(x.shape)
print(y.shape)

print(x)
print(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2)

allAlgorithms = all_estimators(type_filter = 'regressor')

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, '의 정답률 : ', r2_score(y_test, y_pred))
