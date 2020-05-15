# 1. 모듈 임포트
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os

# 2. 데이터 로드
df_train = pd.read_csv('C:/Users/bitcamp/Desktop/서누/'
                       'Kaggle/titanic/train.csv',
                       index_col = 'PassengerId', engine = 'python')
print(df_train.shape)
print(df_train.head(n = 5))


df_test = pd.read_csv('C:/Users/bitcamp/Desktop/서누/'
                      'Kaggle/titanic/test.csv',
                      index_col = 'PassengerId', engine = 'python')
print(df_test.shape)
print(df_test.head(n = 5))


# 3. Exploratory Data Analysis
# 3-1. "Sex" Column
# 3-1-1. countplot
sns.countplot(data = df_train, x = "Sex", hue = "Survived")
plt.show()

# 3-1-2. pivot_table
print(pd.pivot_table(data = df_train, index = "Sex", values = "Survived"))
print("-" * 40)

# 3-2. "Pclass" Column
# 3-2-1. countplot
sns.countplot(data = df_train, x = "Pclass", hue = "Survived")
plt.show()

# 3-2-2. pivot_table
print(pd.pivot_table(data = df_train, index = "Pclass", values = "Survived"))
print("-" * 40)

# 3-3. "Embarked" Column
# 3-3-1. countplot
sns.countplot(data = df_train, x = "Embarked", hue = "Survived")
plt.show()

# 3-3-2. pivot_table
print(pd.pivot_table(data = df_train, index = "Embarked", values = "Survived"))

# 3-4. "Age" & "Fare" Column
# 3-4-1. lmplot
sns.lmplot(data = df_train, x = "Age", y = "Fare", hue = "Survived", fit_reg = False)
plt.show()

# 3-4-2. 이상치 제거 ($500 이상 이상치로 간주)
low_fare = df_train[df_train["Fare"] < 500]
print(low_fare.shape)

# 3-4-3. 이상치가 제거된 데이터로 lmplot 출력
sns.lmplot(data = low_fare, x = "Age", y = "Fare", hue = "Survived", fit_reg = False)
plt.show()

# 3-4-4. $100 이상 이상치로 간주하고 제거
less_than_100 = df_train[df_train["Fare"] < 100]
print(less_than_100.shape)

# 3-4-5. 이상치가 제거된 데이터로 lmplot 출력
sns.lmplot(data = less_than_100, x = "Age", y = "Fare", hue = "Survived", fit_reg = False)
plt.show()

# 3-5. "SibSp" & "Parch"
df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"] + 1
print(df_train.shape)
df_train[["SibSp", "Parch", "FamilySize"]].head(n = 5)

# 시각화
sns.countplot(data = df_train, x = "FamilySize", hue = "Survived")
plt.show()

# 가족 수 변수 만들기
df_train.loc[df_train["FamilySize"] == 1, "FamilyType"] = "Single"     # 단독가구
df_train.loc[(df_train["FamilySize"] > 1) & (df_train["FamilySize"] < 5), "FamilyType"] = "Nuclear"   # 핵가족
df_train.loc[df_train["FamilySize"] >= 5, "FamilyType"] = "Big"    # 대가족
print(df_train.shape)
print(df_train[["FamilySize", "FamilyType"]].head(n = 10))

# 시각화
sns.countplot(data = df_train, x = "FamilyType", hue = "Survived")
plt.show()

pd.pivot_table(data = df_train, index = "FamilyType", values = "Survived")