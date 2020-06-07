# Chapter 08. Pandas 기초
# 8.1 Pandas 개요
# 8.1.1 Series와 DataFrame의 데이터 확인
# [리스트 8-1] Series와 DataFrame의 데이터의 예
import pandas as pd

fruits = {"banana": 3, "orange": 2}
print(pd.Series(fruits))
'''
banana    3
orange    2
dtype: int64
'''

# [리스트 8-2] Series와 DataFrame의 데이터의 예
data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
print(df)
'''
banana    3
orange    2
dtype: int64
       fruits  year  time
0       apple  2001     1
1      orange  2002     4
2      banana  2001     5
3  strawberry  2008     6
4   kiwifruit  2006     3
'''

'''문제
[리스트 8-3]을 실행하여 Series와 DataFrame이
어떤 데이터인지 확인하세요.
'''
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

data = [10, 5, 8, 12, 3]

series = pd.Series(data, index = index)

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)

print("Series 데이터")
print(series)
print("\n")
print("DataFrame 데이터")
print(df)
'''
Series 데이터
apple         10
orange         5
banana         8
strawberry    12
kiwifruit      3
dtype: int64

DataFrame 데이터
       fruits  year  time
0       apple  2001     1
1      orange  2002     4
2      banana  2001     5
3  strawberry  2008     6
4   kiwifruit  2006     3
'''

# 8.2 Series 생성
# 8.2.1 Series 생성
# [리스트 8-5] Series 생성의 예
fruits = {"banana": 3, "orange": 2}
print(pd.Series(fruits))
'''
banana    3
orange    2
dtype: int64
'''

'''문제
pandas를 임포트하세요
데이터에는 data를, 인덱스에는 index를 지정해서
series를 만든 뒤 series에 대입하세요
대문자로 시작하는 Series는 ㄴ데이터형의 이름이며,
소문자로 시작하는 series는 변수명입니다.
'''
import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index = index)
print(series)
'''
apple         10
orange         5
banana         8
strawberry    12
kiwifruit      3
dtype: int64
'''

# 8.2.2 참조
'''문제
인덱스 참조를 사용하여 series의 2~4번째 있는 세 요소를
추출하여 items1에 대입하세요. 인덱스값을 지정하는 방법으로
'apple', 'banana', 'kiwifruit'의 인덱스를 가진 요소를
추출하여 items2에 대입하세요
'''
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index = index)

items1 = series[1:4]

items2 = series[["apple", "banana", "kiwifruit"]]
print(items1)
print()
print(items2)
'''
orange         5
banana         8
strawberry    12
dtype: int64

apple        10
banana        8
kiwifruit     3
dtype: int64
'''

# 8.2.3 데이터와 인덱스 추출
'''문제
변수 series_values에 series의 데이터를 대입
변수 seires_index에 series의 인덱스를 대입
'''
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index = index)

series_values = series.values

series_index = series.index

print(series_values)
print(series_index)
'''
[10  5  8 12  3]
Index(['apple', 'orange', 'banana', 'strawberry', 'kiwifruit'], dtype='object')
'''

# 8.2.4 요소 추가
'''문제
인덱스가 'pineapple'고 데이터가 12인 요소를 series에 추가하세요
'''
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index = index)

pineapple = pd.Series([12], index = ["pineapple"])
series = series.append(pineapple)
print(series)
'''
apple         10
orange         5
banana         8
strawberry    12
kiwifruit      3
pineapple     12
dtype: int64
'''

# 8.2.5 요소 삭제
'''문제
인덱스가 strawberry인 요소를 삭제한 Series형 변수를 series에 대입
'''
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index = index)

series = series.drop("strawberry")

print(series)
'''
apple        10
orange        5
banana        8
kiwifruit     3
dtype: int64
'''

# 8.2.6 필터링
'''문제
series의 요소 중에서 5 이상 10 미만의 요소를 포함하는
Series를 만들어 series에 다시 대입하세요
'''
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index = index)

series = series[series >= 5][series < 10]

print(series)
'''
orange    5
banana    8
dtype: int64
'''

# 8.2.7 정렬
'''문제
series의 인덱스를 알파벳 순으로 정렬해서 items1에 대입
series의 데이터값을 오름차순으로 정렬해서 items2에 대입
'''
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index = index)

items1 = series.sort_index()

items2 = series.sort_values()

print(items1)
print()
print(items2)
'''
apple         10
banana         8
kiwifruit      3
orange         5
strawberry    12
dtype: int64

kiwifruit      3
orange         5
banana         8
apple         10
strawberry    12
dtype: int64
'''

# 8.3 DataFrame
# 8.3.1 DataFrame 생성
'''문제
series1과 series2로 DataFrame을 생성하여 df에 대입
'''
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index = index)
series2 = pd.Series(data2, index = index)

df = pd.DataFrame([series1, series2])

print(df)
'''
   apple  orange  banana  strawberry  kiwifruit
0     10       5       8          12          3
1     30      25      12          10          8
'''

# 8.3.2 인덱스와 컬럼 설정
'''문제
DataFrame형의 변수 df의 인덱스가 1부터 시작하도록 설정
'''
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index = index)
series2 = pd.Series(data2, index = index)
df = pd.DataFrame([series1, series2])

df.index = [1, 2]

print(df)
'''
   apple  orange  banana  strawberry  kiwifruit
1     10       5       8          12          3
2     30      25      12          10          8
'''

# 8.3.3 행 추가
'''문제
DataFrame형의 변수 df에 새로운 행으로 series3을 추가
DataFrame의 컬럼과 추가할 Series의 인덱스가 일치하지 않을 때의 동작을 확인
'''
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
data3 = [30, 12, 10, 8, 25, 3]
series1 = pd.Series(data1, index = index)
series2 = pd.Series(data2, index = index)

index.append("pineapple")
series3 = pd.Series(data3, index = index)
df = pd.DataFrame([series1, series2])

df = df.append(series3, ignore_index = True)

print(df)
'''
   apple  orange  banana  strawberry  kiwifruit  pineapple
0     10       5       8          12          3        NaN
1     30      25      12          10          8        NaN
2     30      12      10           8         25        3.0
'''

# 8.3.4 열 추가
'''문제
df에 새로운 열 'mango'를 만들어 new_column의 데이터를 추가
'''
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index = index)
series2 = pd.Series(data2, index = index)

new_column = pd.Series([15, 7], index = [0, 1])

df = pd.DataFrame([series1, series2])

df["mango"] = new_column

print(df)
'''
   apple  orange  banana  strawberry  kiwifruit  mango
0     10       5       8          12          3     15
1     30      25      12          10          8      7
'''

# 8.3.5 데이터 참조
'''문제
loc[]을 사용하여 df의 2행부터 5행까지의 4행과 'banana', 'kiwifruit'의 2열을 포함한
DataFrame을 df에 대입, 첫번째 행의 인덱스는 1이며, 이후의 인덱스는 정수의 오름차순
'''
import numpy as np
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
    
df.index = range(1, 11)

df = df.loc[range(2,6),["banana","kiwifruit"]]

print(df)
'''
   banana  kiwifruit
2      10         10
3       9          1
4      10          5
5       5          8
'''

'''문제
iloc[]을 사용하여 df의 2행부터 5행까지의 4행과 'banana', 'kiwifruit'의 2열을 포함한
DataFrame을 df에 대입하세요
'''
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

df = df.iloc[range(1,5), [2, 4]] # 슬라이스를 사용하여 df = df.iloc[1:5, [2,4]] 도 가능합니다 

print(df)
'''
   banana  kiwifruit
2      10         10
3       9          1
4      10          5
5       5          8
'''

# 8.3.6 행 또는 열 삭제
'''문제
drop()을 이용하여 df에서 홀수 인덱스가 붙은 행만 남기고 df에 대입
drop()을 이용하여 df의 열 'strawberry'를 삭제하고 df에 대입
'''
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

df = df.drop(np.arange(2, 11, 2))
df = df.drop("strawberry", axis = 1) 

print(df)
'''
   apple  orange  banana  kiwifruit
1      6       8       6         10
3      4       9       9          1
5      8       2       5          8
7      4       8       1          3
9      3       9       6          3
'''

# 8.3.7 정렬
'''문제
df를 'apple', 'orange', 'banana', 'strawberry', 'kiwifruit' 순으로
오름차순 정렬, 정렬한 결과로 만들어진 DataFrame을 df에 대입
'''
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

df = df.sort_values(by = columns)

print(df)
'''
    apple  orange  banana  strawberry  kiwifruit
2       1       7      10           4         10
9       3       9       6           1          3
7       4       8       1           4          3
3       4       9       9           9          1
4       4       9      10           2          5
10      5       2       1           2          1
8       6       8       4           8          8
1       6       8       6           3         10
5       8       2       5           4          8
6      10       7       4           4          4
'''

# 8.3.8 필터링
'''문제
필터링을 사용하여 df의 'apple' 열이 5 이상, 'kiwifruit' 열이 5 이상의 값을 가진 행을
포함한 DataFrame을 df에 대입
'''
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

df = df.loc[df["apple"] >= 5]
df = df.loc[df["kiwifruit"] >= 5]

print(df)
'''
   apple  orange  banana  strawberry  kiwifruit
1      6       8       6           3         10
5      8       2       5           4          8
8      6       8       4           8          8
'''

'''연습 문제'''
# 문제 [리스트 8-53]의 주석에서 언급하는 처리를 구현하세요
index = ["growth", "mission", "ishikawa", "pro"]
data = [50, 7, 26, 1]

series = pd.Series(data, index=index)

aidemy = series.sort_index()

aidemy1 = pd.Series([30], index=["tutor"])
aidemy2 = series.append(aidemy1)

print(aidemy)
print()
print(aidemy2)

df = pd.DataFrame()
for index in index:
    df[index] = np.random.choice(range(1, 11), 10)

df.index = range(1, 11)

aidemy3 = df.loc[range(2,6),["ishikawa"]]
print()
print(aidemy3)
'''
growth      50
ishikawa    26
mission      7
pro          1
dtype: int64

growth      50
mission      7
ishikawa    26
pro          1
tutor       30
dtype: int64

   ishikawa
2         4
3         6
4        10
5         5
'''