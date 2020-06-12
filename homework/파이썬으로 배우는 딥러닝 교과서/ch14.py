# Chapter 14. DataFrame을 이용한 데이터 클렌징
# 14.1 CSV
# 14.1.1 Pandas로 CSV 읽기
# [리스트 14-1] Pandas로 CSV를 읽는 예
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)
df.columns = ['', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
              '0D280/0D315 of diluted wines', 'Proline']
print(df)

'''문제
다음 웹사이트에서 붓꽃 데이터를 CSV 형식으로 불러들이고, Pandas의 DataFrame형으로 출력하세요
컬럼은 왼쪽부터 'sepal length', 'sepal width', 'petal length', 'petal width', 'Class'를 지정하세요
'''
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'Class']
print(iris)


# 14.1.2 CSV 라이브러리로 CSV 만들기
# [리스트 14-4] CSV 라이브러리로 CSV를 작성하는 예
import csv

# with 문을 사용해 파일을 처리합니다
with open("csv0.csv", "w") as csvfile:
    # writer() 메서드의 인수로 csvfile과 개행(줄바꿈) 코드(\n)를 지정합니다
    writer = csv.writer(csvfile, lineterminator="\n")

    # writerow(리스트) 로 행을 추가합니다
    writer.writerow(["city", "year", "season"])
    writer.writerow(["Nagano", 1998, "winter"])
    writer.writerow(["Sydney", 2000, "summer"])
    writer.writerow(["Salt Lake City", 2002, "winter"])
    writer.writerow(["Athens", 2004, "summer"])
    writer.writerow(["Torino", 2006, "winter"])
    writer.writerow(["Beijing", 2008, "summer"])
    writer.writerow(["Vancouver", 2010, "winter"])
    writer.writerow(["London", 2012, "summer"])
    writer.writerow(["Sochi", 2014, "winter"])
    writer.writerow(["Rio de Janeiro", 2016, "summer"])
    
# 14.1.3 Pandas로 CSV 만들기
# data = {'city': ['Nagano', 'Sydney', 'Salt Lake City', 'Athens', 'Torino', 'Beijing',
#                  'Vancouver', 'London', 'Sochi', 'Rio de Janeiro'],
#         'year': ['winter', 'summer', 'winter', 'summer', 'winter', 'summer','winter',
#                  'summer', 'winter', 'summer']}
# df = pd.DataFrame(data)
# df.to_csv('csv1.csv')

# 문제
# data = {"OS": ["Machintosh", "Windows", "Linux"],
#         "release": [1984, 1985, 1991],
#         "country": ["US", "US", ""]}
# df = pd.DataFrame(data)
# df.to_csv("OSlist.csv")

'''문제
attri_data_frame1에 attri_data_frame2 행을 추가해서 출력하세요
'''
from pandas import Series, DataFrame
attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name": ["영이", "순돌", "짱구", "태양", "션",
                        "유리", "현아", "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)
attri_data2 = {"ID": ["107", "109"],
               "city": ["봉화", "전주"],
               "birth_year": [1994, 1988]}

attri_data_frame2 = DataFrame(attri_data2)

attri_data_frame1.append(attri_data_frame2).sort_values(by = "ID", ascending = True).reset_index(drop = True)
print(attri_data_frame1)

# 14.3 결측치
# 14.3.1 리스트와이즈 삭제와 페어와이즈 삭제
# [리스트 14-11] 테이블의 일부를 누락시킨 예
import numpy as np
from numpy import nan as NA

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

# 일부 데이터를 일부러 누락시킴
sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA 
print(sample_data_frame)

# [리스트 14-12] 리스트와이즈 삭제의 예
sample_data_frame = sample_data_frame.dropna()
print(sample_data_frame)

# [리스트 14-13] 페어와이즈 삭제의 예
sample_data_frame = sample_data_frame[[0,1,2]].dropna()
print(sample_data_frame)

'''문제
DataFrame의 0열과 2열을 남기되 NaN을 포함하는 행은 삭제하고 출력하세요
'''
np.random.seed(0)

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA
sample_data_frame = sample_data_frame[[0, 2]].dropna()
print(sample_data_frame)

# 14.3.2 결측치 보완
# [리스트 14-16] 결측치 보완의 예(1)
sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA
print(sample_data_frame)

# [리스트 14-17] 결측치 보완의 예(2)
sample_data_frame = sample_data_frame.fillna(0)
print(sample_data_frame)

# [리스트 14-18] 결측치 보완의 예(3)
sample_data_frame = sample_data_frame.fillna(method = 'ffill')
print(sample_data_frame)

'''문제
DataFrame의 NaN 부분을 앞에 있는 데이터로 채워서 출력하세요
'''
sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[6:, 2] = NA

sample_data_frame = sample_data_frame.fillna(method = "ffill")
print(sample_data_frame)

# 14.3.3 결측치 보완(평균값 대입법)
# [리스트 14-21] 결측치 보완의 예
sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA

sample_data_frame = sample_data_frame.fillna(sample_data_frame.mean())
print(sample_data_frame)

'''문제
DataFrame의 NaN 부분을 열의 평균값으로 채워서 출력하세요
'''
sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[6:, 2] = NA

sample_data_frame = sample_data_frame.fillna(sample_data_frame.mean())
print(sample_data_frame)

# 14.4 데이터 요약
# 14.4.1 키별 통계량 산출
# [리스트 14-24] 키별 통계량 산출의 예
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols", 
            "Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
print(df["Alcohol"].mean())         # 13.000617977528083

'''문제
와인 데이터셋에서 Magnesium의 평균값을 출력하세요
'''
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols", 
            "Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
print(df["Magnesium"].mean())       # 99.74157303370787

# 14.4.2 중복 데이터
# [리스트 14-27] 중복 데이터의 예(1)
dupli_data = DataFrame({"col1":[1, 1, 2, 3, 4, 4, 6, 6], 
                        "col2":["a", "b", "b", "b", "c", "c", "b", "b"]}) 
print(dupli_data)
'''
   col1 col2
0     1    a
1     1    b
2     2    b
3     3    b
4     4    c
5     4    c
6     6    b
7     6    b
'''
# [리스트 14-28] 중복 데이터의 예(2)
print(dupli_data.duplicated())
'''
0    False
1    False
2    False
3    False
4    False
5     True
6    False
7     True
dtype: bool
'''

# [리스트 14-29] 중복 데이터의 예(3)
print(dupli_data.drop_duplicates())
'''
   col1 col2
0     1    a
1     1    b
2     2    b
3     3    b
4     4    c
6     6    b
'''

'''문제
다음 DataFrame에는 중복된 데이터가 들어 있습니다. 이를 삭제하고 새로운 DataFrame을 출력하세요
'''
dupli_data = DataFrame({"col1":[1, 1, 2, 3, 4, 4, 6, 6, 7, 7, 7, 8, 9, 9],
                        "col2":["a", "b", "b", "b", "c", "c", "b", "b", "d", "d", "c", "b", "c", "c"]})
print(dupli_data.drop_duplicates())
'''
    col1 col2
0      1    a
1      1    b
2      2    b
3      3    b
4      4    c
6      6    b
8      7    d
10     7    c
11     8    b
12     9    c
'''

# 14.4.3 매핑
# 매핑은 공통의 키 역할을 하는 데이터의 값을 가져오는 처리
# [리스트 14-32] 매핑의 예(1)
attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name": ["영이", "순돌", "짱구", "태양", "션",
                        "유리", "현아", "태식", "민수", "호식"]}
print(attri_data_frame1)
'''
    ID city  birth_year name
0  100   서울        1990   영이
1  101   부산        1989   순돌
2  102   대전        1992   짱구
3  103   광주        1997   태양
4  104   서울        1982    션
5  106   서울        1991   유리
6  108   부산        1988   현아
7  110   대전        1990   태식
8  111   광주        1995   민수
9  113   서울        1981   호식
'''

# [리스트 14-33] 매핑의 예(2)
city_map ={"서울":"서울", 
           "광주":"전라도", 
           "부산":"경상도", 
           "대전":"충청도"}
print(city_map)
'''
{'서울': '서울', '광주': '전라도', '부산': '경상도', '대전': '충청도'}
'''

# [리스트 14-34] 매핑의 예(2)
attri_data_frame1["region"] = attri_data_frame1["city"].map(city_map)
print(attri_data_frame1)
'''
0  100   서울        1990   영이     서울
1  101   부산        1989   순돌    경상도
2  102   대전        1992   짱구    충청도
3  103   광주        1997   태양    전라도
4  104   서울        1982    션     서울
5  106   서울        1991   유리     서울
6  108   부산        1988   현아    경상도
7  110   대전        1990   태식    충청도
8  111   광주        1995   민수    전라도
9  113   서울        1981   호식     서울
'''

'''문제
다음의 DataFrame에서 city가 서울이나 대전이면 '중부', 광주나 부산이면 '남부'가 되도록
새 열(MS)을 추가하고, 결과를 출력하세요
'''
attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name": ["영이", "순돌", "짱구", "태양", "션",
                        "유리", "현아", "태식", "민수", "호식"]}
WE_map = {"서울": "중부",
          "광주": "남부",
          "부산": "남부",
          "대전": "중부"}

attri_data_frame1["MS"] = attri_data_frame1["city"].map(WE_map)
print(attri_data_frame1)
'''
    ID city  birth_year name region  MS
0  100   서울        1990   영이     서울  중부
1  101   부산        1989   순돌    경상도  남부
2  102   대전        1992   짱구    충청도  중부
3  103   광주        1997   태양    전라도  남부
4  104   서울        1982    션     서울  중부
5  106   서울        1991   유리     서울  중부
6  108   부산        1988   현아    경상도  남부
7  110   대전        1990   태식    충청도  중부
8  111   광주        1995   민수    전라도  남부
9  113   서울        1981   호식     서울  중부
'''

# 14.4.4 구간 분할
# [리스트 14-37] 구간 분할의 예(1)
attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name": ["영이", "순돌", "짱구", "태양", "션",
                        "유리", "현아", "태식", "민수", "호식"]}
attri_data_frame1 = DataFrame(attri_data1)

# [리스트 14-38] 구간 분할의 예(2)
birth_year_bins = [1980, 1985, 1990, 1995, 2000]
birth_year_cut_data = pd.cut(attri_data_frame1.birth_year, birth_year_bins)
print(birth_year_cut_data)
'''
0    (1985, 1990]
1    (1985, 1990]
2    (1990, 1995]
3    (1995, 2000]
4    (1980, 1985]
5    (1990, 1995]
6    (1985, 1990]
7    (1985, 1990]
8    (1990, 1995]
9    (1980, 1985]
Name: birth_year, dtype: category
Categories (4, interval[int64]): [(1980, 1985] < (1985, 1990] < (1990, 1995] < (1995, 2000]]
'''

# [리스트 14-39] 구간 분할의 예(3)
print(pd.value_counts(birth_year_cut_data))
'''
(1985, 1990]    4
(1990, 1995]    3
(1980, 1985]    2
(1995, 2000]    1
Name: birth_year, dtype: int64
'''

# [리스트 14-40] 구간 분할의 예(4)
group_names = ["first1980", "second1980", "first1990", "second1990"]
birth_year_cut_data = pd.cut(attri_data_frame1.birth_year,birth_year_bins,labels = group_names)
print(pd.value_counts(birth_year_cut_data))
'''
second1980    4
first1990     3
first1980     2
second1990    1
Name: birth_year, dtype: int64
'''

# [리스트 14-41] 구간 분할의 예(5)
print(pd.cut(attri_data_frame1.birth_year, 2))
'''
0      (1989.0, 1997.0]
1    (1980.984, 1989.0]
2      (1989.0, 1997.0]
3      (1989.0, 1997.0]
4    (1980.984, 1989.0]
5      (1989.0, 1997.0]
6    (1980.984, 1989.0]
7      (1989.0, 1997.0]
8      (1989.0, 1997.0]
9    (1980.984, 1989.0]
Name: birth_year, dtype: category
Categories (2, interval[float64]): [(1980.984, 1989.0] < (1989.0, 1997.0]]
'''

'''문제
다음의 DataFrame의 ID를 두 구간으로 분할해서 출력하세요
'''
attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name": ["영이", "순돌", "짱구", "태양", "션",
                        "유리", "현아", "태식", "민수", "호식"]}
attri_data_frame1 = DataFrame(attri_data1)
# print(pd.cut(attri_data_frame1.ID, 2))


'''연습문제'''
# 문제 다음에서 주석 부분의 코드를 작성하세요
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)

# 각각의 수치가 나타내는 바를 컬럼에 추가합니다
df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash",
            "Magnesium", "Total phenols", "Flavanoids",
            "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
            "OD280/OD315 of diluted wines","Proline"]

# 변수 df의 상위 10행을 변수 df_ten에 대입하여 표시하세요
df_ten = df.head(10)
print(df_ten)

# 데이터의 일부를 누락시킵니다
df_ten.iloc[1,0] = NA
df_ten.iloc[2,3] = NA
df_ten.iloc[4,8] = NA
df_ten.iloc[7,3] = NA
print(df_ten)

# fillna() 메서드로 NaN 부분에 열의 평균값을 대입하세요
df_ten.fillna(df_ten.mean())
print(df_ten)

# "Alcohol" 열의 평균을 출력하세요
print(df_ten["Alcohol"].mean())

# 중복된 행을 제거하세요
df_ten.append(df_ten.loc[3])
df_ten.append(df_ten.loc[6])
df_ten.append(df_ten.loc[9])
df_ten = df_ten.drop_duplicates()
print(df_ten)

# Alcohol 열의 구간 리스트를 작성하세요
alcohol_bins = [0,5,10,15,20,25]
alcoholr_cut_data = pd.cut(df_ten["Alcohol"],alcohol_bins)

# 구간 수를 집계하여 출력하세요
print(pd.value_counts(alcoholr_cut_data))