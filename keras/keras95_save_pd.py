import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/iris.csv", index_col = None, header = 0, sep = ',')
'''
index_col = 맨 앞 컬럼을 인덱스로 사용할지?
header = 맨 위 행을 헤더로 쓸지?
'''
print(datasets)

print(datasets.head(n = 5))         # 데이터의 상단 n개 행 출력
print("=" * 35)
print(datasets.tail(n = 5))         # 데이터의 하단 n개 행 출력
print("=" * 35)
print(datasets.values)              # pandas data를 numpy 형식으로 바꾸어준다. -> 매우 유용!! 자주 씀 !!

aaa = datasets.values
print(type(aaa))                    # <class 'numpy.ndarray'>


np.save('./data/csv/iris.npy', arr = aaa)