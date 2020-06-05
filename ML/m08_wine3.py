import pandas as pd
import matplotlib.pyplot as plt

# 와인 데이터 읽기
wine = pd.read_csv('./data/csv/winequality-white.csv',
                   sep = ';', header = 0)

count_data = wine.groupby('quality')['quality'].count()
# count() 함수는 개체 별로 숫자를 세줌
# quality에 있는 10개의 등급이 각각 몇개씩 있는지
# column에 있는 개체의 수
# groupby() 함수는 column의 개체를 묶어줌

print(count_data)
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
Name: quality, dtype: int64
'''

count_data.plot()
plt.show()