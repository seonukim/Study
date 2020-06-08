# Chapter 09. Pandas 응용
# 9.1 DataFrame 연결과 결합의 개요
# 9.2 DataFrame 연결
'''문제
DataFrame 변수 df_data1과 df_data2를 세로로 연결하여 df1에 대입
DataFrame 변수 df_data1과 df_data2를 가로로 연결하여 df2에 대입
'''
import numpy as np
import pandas as pd

def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1, 101), len(index))
    df.index = index
    return df

columns = ["apple", "orange", "banana"]
df_data1 = make_random_df(range(1, 5), columns, 0)
df_data2 = make_random_df(range(1, 5), columns, 1)

df1 = pd.concat([df_data1, df_data2], axis = 0)

df2 = pd.concat([df_data1, df_data2], axis = 1)

print(df1)
print(df2)
'''
   apple  orange  banana
1     45      68      37
2     48      10      88
3     65      84      71
4     68      22      89
1     38      76      17
2     13       6       2
3     73      80      77
4     10      65      72
   apple  orange  banana  apple  orange  banana
1     45      68      37     38      76      17
2     48      10      88     13       6       2
3     65      84      71     73      80      77
4     68      22      89     10      65      72
'''

# 9.2.2 인덱스나 컬럼이 일치하지 않는 DataFrame 간의 연결
'''문제
인덱스나 컬럼이 일치하지 않는 DataFrame끼리 연결했을 때 어떻게 동작하는지 확인
DataFrame형 변수 df_data1과 df_data2를 세로로 연결하여 df1에, DataFrame형 변수
df_data1과 df_data2를 가로로 연결하여 df2에 대입
'''
columns1 = ["apple", "orange", "banana"]
columns2 = ["orange", "kiwifruit", "banana"]

df_data1 = make_random_df(range(1, 5), columns1, 0)

df_data2 = make_random_df(np.arange(1, 8, 2), columns2, 1)

df1 = pd.concat([df_data1, df_data2], axis = 0)

df2 = pd.concat([df_data1, df_data2], axis = 1) 

print(df1)
print(df2)
'''
   apple  orange  banana  kiwifruit
1   45.0      68      37        NaN
2   48.0      10      88        NaN
3   65.0      84      71        NaN
4   68.0      22      89        NaN
1    NaN      38      17       76.0
3    NaN      13       2        6.0
5    NaN      73      77       80.0
7    NaN      10      72       65.0
   apple  orange  banana  orange  kiwifruit  banana
1   45.0    68.0    37.0    38.0       76.0    17.0
2   48.0    10.0    88.0     NaN        NaN     NaN
3   65.0    84.0    71.0    13.0        6.0     2.0
4   68.0    22.0    89.0     NaN        NaN     NaN
5    NaN     NaN     NaN    73.0       80.0    77.0
7    NaN     NaN     NaN    10.0       65.0    72.0
'''

# 9.2.3 연결 시 라벨 지정하기
'''문제
DataFrame형 변수 df_data1과 df_data2를 가로로 연결하고, keys로 "X"와 "Y"를 지정하여
MultiIndex로 만든 뒤 df에 대입
df의 "Y" 라벨 'banana'를 Y_banana에 대입
'''
columns = ["apple", "orange", "banana"]
df_data1 = make_random_df(range(1, 5), columns, 0)
df_data2 = make_random_df(range(1, 5), columns, 1)

df = pd.concat([df_data1, df_data2], axis = 1, keys = ["X", "Y"])

Y_banana = df["Y", "banana"]

print(df)
print()
print(Y_banana)
'''
      X                   Y
  apple orange banana apple orange banana
1    45     68     37    38     76     17
2    48     10     88    13      6      2
3    65     84     71    73     80     77
4    68     22     89    10     65     72

1    17
2     2
3    77
4    72
Name: (Y, banana), dtype: int32
'''

# 9.3 DataFrame 결합
# 9.3.1 결합 유형
# 9.3.2 내부 결합의 기본
'''문제
내부 결합의 동작을 확인하세요
DataFrame형 변수 df1과 df2의 컬럼 'fruits'를 Keys로 하여 내부 결합한
DataFrame을 df3에 대입하세요
'''
data1 = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
         "year": [2001, 2002, 2001, 2008, 2006],
         "amount": [1, 4, 5, 6, 3]}
df1 = pd.DataFrame(data1)

data2 = {"fruits": ["apple", "orange", "banana", "strawberry", "mango"],
         "year": [2001, 2002, 2001, 2008, 2007],
         "price": [150, 120, 100, 250, 3000]}
df2 = pd.DataFrame(data2)

print(df1)
print()
print(df2)
print()
df3 = pd.merge(df1, df2, on = "fruits", how = "inner")

print(df3)
'''
       fruits  year  amount
0       apple  2001       1
1      orange  2002       4
2      banana  2001       5
3  strawberry  2008       6
4   kiwifruit  2006       3

       fruits  year  price
0       apple  2001    150
1      orange  2002    120
2      banana  2001    100
3  strawberry  2008    250
4       mango  2007   3000

       fruits  year_x  amount  year_y  price
0       apple    2001       1    2001    150
1      orange    2002       4    2002    120
2      banana    2001       5    2001    100
3  strawberry    2008       6    2008    250
'''

# 9.3.3 외부 결합의 기본
'''문제
외부 결합의 동작을 확인하세요
DataFrame형 변수 df1과 df2의 컬럼 'fruits'를 Keys로 하여
외부 결합한 DataFrame을 df3에 대입하세요
'''
data1 = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
         "year": [2001, 2002, 2001, 2008, 2006],
         "amount": [1, 4, 5, 6, 3]}

df1 = pd.DataFrame(data1)

data2 = {"fruits": ["apple", "orange", "banana", "strawberry", "mango"],
         "year": [2001, 2002, 2001, 2008, 2007],
         "price": [150, 120, 100, 250, 3000]}
df2 = pd.DataFrame(data2)

print(df1)
print()
print(df2)
print()

df3 = pd.merge(df1, df2, on = "fruits", how = "outer")

print(df3)
'''
       fruits  year  amount
0       apple  2001       1
1      orange  2002       4
2      banana  2001       5
3  strawberry  2008       6
4   kiwifruit  2006       3

       fruits  year  price
0       apple  2001    150
1      orange  2002    120
2      banana  2001    100
3  strawberry  2008    250
4       mango  2007   3000

       fruits  year_x  amount  year_y   price
0       apple  2001.0     1.0  2001.0   150.0
1      orange  2002.0     4.0  2002.0   120.0
2      banana  2001.0     5.0  2001.0   100.0
3  strawberry  2008.0     6.0  2008.0   250.0
4   kiwifruit  2006.0     3.0     NaN     NaN
5       mango     NaN     NaN  2007.0  3000.0
'''

# 9.3.4 이름이 다른 열을 Key로 결합하기
'''문제
주문 정보(order_df)와 고객 정보(customer_df)를 결합하세요
order_df의 'customer_id'로 customer_df의 'id'를 참조하세요
결합 방식은 내부결합
'''
order_df = pd.DataFrame([[1000, 2546, 103],
                         [1001, 4352, 101],
                         [1002, 342, 101]],
                        columns = ["id", "item_id", "customer_id"])

customer_df = pd.DataFrame([[101, "광수"],
                            [102, "민호"],
                            [103, "소희"]],
                           columns = ["id", "name"])

order_df = pd.merge(order_df, customer_df,
                    left_on = "customer_id",
                    right_on = "id", how = "inner")

print(order_df)
'''
   id_x  item_id  customer_id  id_y name
0  1000     2546          103   103   소희
1  1001     4352          101   101   광수
2  1002      342          101   101   광수
'''

# 9.3.5 인덱스를 Key로 결합하기
'''문제
주문 정보(order_df)와 고객 정보(customer_df)를 결합하세요
order_df의 'customer_id'로 customer_df의 인덱스를 참조하세요
결합 방식은 내부 결합
'''
order_df = pd.DataFrame([[1000, 2546, 103],
                         [1001, 4352, 101],
                         [1002, 342, 101]],
                        columns = ["id", "item_id", "customer_id"])

customer_df = pd.DataFrame([["광수"],
                            ["민호"],
                            ["소희"]],
                           columns = ["name"])

customer_df.index = [101, 102, 103]
order_df = pd.merge(order_df, customer_df,
                    left_on = "customer_id",
                    right_index = True, how = "inner")

print(order_df)
'''
     id  item_id  customer_id name
0  1000     2546          103   소희
1  1001     4352          101   광수
2  1002      342          101   광수
'''

# 9.4 DataFrame을 이용한 데이터 분석
# 9.4.1 특정 행 얻기
'''문제
DataFrame형 변수 df의 첫 3행을 취득하여 df_head에 대입
DataFrame형 변수 df의 끝 3행을 취득하여 df_tail에 대입
'''
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

df_head = df.head(3)

df_tail = df.tail(3)

print(df_head)
print(df_tail)
'''
   apple  orange  banana  strawberry  kiwifruit
1      6       8       6           3         10
2      1       7      10           4         10
3      4       9       9           9          1
    apple  orange  banana  strawberry  kiwifruit
8       6       8       4           8          8
9       3       9       6           1          3
10      5       2       1           2          1
'''

# 9.4.2 계산 처리하기
'''문제
df의 각 요소를 두 배로 만들어 double_df에 대입
df의 각 요소를 제곱하여 square_df에 대입
df의 각 요소의 제곱근을 계산하여 sqrt_df에 대입
'''
import math
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

double_df = df * 2 # double_df = df + df도 OK입니다

square_df = df * df #square_df = df**2도 OK입니다

sqrt_df = np.sqrt(df) 

print(double_df)
print(square_df)
print(sqrt_df)
'''
    apple  orange  banana  strawberry  kiwifruit
1      12      16      12           6         20
2       2      14      20           8         20
3       8      18      18          18          2
4       8      18      20           4         10
5      16       4      10           8         16
6      20      14       8           8          8
7       8      16       2           8          6
8      12      16       8          16         16
9       6      18      12           2          6
10     10       4       2           4          2
    apple  orange  banana  strawberry  kiwifruit
1      36      64      36           9        100
2       1      49     100          16        100
3      16      81      81          81          1
4      16      81     100           4         25
5      64       4      25          16         64
6     100      49      16          16         16
7      16      64       1          16          9
8      36      64      16          64         64
9       9      81      36           1          9
10     25       4       1           4          1
       apple    orange    banana  strawberry  kiwifruit
1   2.449490  2.828427  2.449490    1.732051   3.162278
2   1.000000  2.645751  3.162278    2.000000   3.162278
3   2.000000  3.000000  3.000000    3.000000   1.000000
4   2.000000  3.000000  3.162278    1.414214   2.236068
5   2.828427  1.414214  2.236068    2.000000   2.828427
6   3.162278  2.645751  2.000000    2.000000   2.000000
7   2.000000  2.828427  1.000000    2.000000   1.732051
8   2.449490  2.828427  2.000000    2.828427   2.828427
9   1.732051  3.000000  2.449490    1.000000   1.732051
10  2.236068  1.414214  1.000000    1.414214   1.000000
'''

# 9.4.3 통계 정보 얻기
'''문제
DataFrame형 변수 df의 통계 정보 중 'mean', 'max', 'min'을 꺼내서 df_des에 대입
'''
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

df_des = df.describe().loc[["mean", "max", "min"]]

print(df_des)
'''
      apple  orange  banana  strawberry  kiwifruit
mean    5.1     6.9     5.6         4.1        5.3
max    10.0     9.0    10.0         9.0       10.0
min     1.0     2.0     1.0         1.0        1.0
'''

# 9.4.4 DataFrame의 행간 차이와 열간 차이 구하기
'''문제
DataFrame형 변수 df의 각 행에 대해 2행 뒤와의 차이를 계산한 DataFrame을 df_diff에 대입
'''
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

df_diff = df.diff(-2, axis = 0)

print(df)
print(df_diff)
'''
    apple  orange  banana  strawberry  kiwifruit
1       6       8       6           3         10
2       1       7      10           4         10
3       4       9       9           9          1
4       4       9      10           2          5
5       8       2       5           4          8
6      10       7       4           4          4
7       4       8       1           4          3
8       6       8       4           8          8
9       3       9       6           1          3
10      5       2       1           2          1
    apple  orange  banana  strawberry  kiwifruit
1     2.0    -1.0    -3.0        -6.0        9.0
2    -3.0    -2.0     0.0         2.0        5.0
3    -4.0     7.0     4.0         5.0       -7.0
4    -6.0     2.0     6.0        -2.0        1.0
5     4.0    -6.0     4.0         0.0        5.0
6     4.0    -1.0     0.0        -4.0       -4.0
7     1.0    -1.0    -5.0         3.0        0.0
8     1.0     6.0     3.0         6.0        7.0
9     NaN     NaN     NaN         NaN        NaN
10    NaN     NaN     NaN         NaN        NaN
'''

# 9.4.5 그룹화
'''문제
DataFrame형 변수 prefecture_df는 도시 이름, 면적(정수값), 인구(정수),
지역명을 포함하고 있습니다. prefecture_df를 지역으로 그룹화하여
grouped_region에 대입
prefecture_df의 지역별 면적과 인구의 평균을 mean_df에 대입하세요
'''
# 도시 정보
prefecture_df = pd.DataFrame([["강릉", 1040, 213527, "강원도"], 
                              ["광주", 430, 1458915, "전라도"],
                              ["평창", 1463, 42218, "강원도"],
                              ["대전", 539, 1476955, "충청도"],
                              ["단양", 780, 29816, "충청도"]],
                             columns = ["Prefecture", "Area",
                                      "Population", "Region"])
print(prefecture_df)
'''
  Prefecture  Area  Population Region
0         강릉  1040      213527    강원도
1         광주   430     1458915    전라도
2         평창  1463       42218    강원도
3         대전   539     1476955    충청도
4         단양   780       29816    충청도
'''

grouped_region = prefecture_df.groupby("Region")
mean_df = grouped_region.mean()
print(mean_df)
'''
          Area  Population
Region
강원도     1251.5    127872.5
전라도      430.0   1458915.0
충청도      659.5    753385.5
'''

'''연습 문제'''
# 문제 df1와 df2는 각각 야채와 과일에 대한 DataFrame이다
# "Name", "Type", "Price"는 각각 이름, 종류, 가격을 나타낸다
# 여러분은 야채와 과일을 각각 3개씩 구입할 생각이다
# 가급적 저렴하게 구매하려 하므로 다음 순서로 최소 비용을 구하세요
# df1와 df2를 세로로 결합한다
# 야채와 과일을 각각 추출하여 'Price'로 정렬한다
# 야채와 과일을 저렴한 순으로 위에서 세 개씩을 선택하여 총액을 계산하고 출력한다
df1 = pd.DataFrame([["apple", "Fruit", 120],
                    ["orange", "Fruit", 60],
                    ["banana", "Fruit", 100],
                    ["pumpkin", "Vegetable", 150],
                    ["potato", "Vegetable", 80]],
                    columns = ["Name", "Type", "Price"])

df2 = pd.DataFrame([["onion", "Vegetable", 60],
                    ["carrot", "Vegetable", 50],
                    ["beans", "Vegetable", 100],
                    ["grape", "Fruit", 160],
                    ["kiwifruit", "Fruit", 80]],
                   columns = ["Name", "Type", "Price"])

df3 = pd.concat([df1, df2], axis=0)

df_fruit = df3.loc[df3["Type"] == "Fruit"]
df_fruit = df_fruit.sort_values(by="Price")

df_veg = df3.loc[df3["Type"] == "Vegetable"]
df_veg = df_veg.sort_values(by = "Price")

print(sum(df_fruit[:3]["Price"]) + sum(df_veg[:3]["Price"]))    # 430

'''종합 문제'''
# 문제 [리스트 9-27]의 DataFrame에 대해 주석에서 말하는 처리를 구현하세요
index = ["광수", "민호", "소희", "태양", "영희"]
columns = ["국어", "수학", "사회", "과학", "영어"]
data = [[30, 45, 12, 45, 87], [65, 47, 83, 17, 58],
        [64, 63, 86, 57, 46,], [38, 47, 62, 91, 63], [65, 36, 85, 94, 36]]
df = pd.DataFrame(data, index = index, columns = columns)

# df에 "체육"이라는 새 열을 만들어 pe_column의 데이터를 추가하세요
pe_column = pd.Series([56, 43, 73, 82, 62],
                      index = ["광수", "민호", "소희", "태양", "영희"])
df["체육"] = pe_column
print(df)
print()

# 수학을 오름차순으로 정렬합니다
df1 = df.sort_values(by = "수학", ascending = True)
print(df1)
print()

# df1의 각 요소에 5점을 더하세요
df2 = df1 + 5
print(df2)
print()

# df의 통계 정보 중에서 "mean", "max", "min"을 출력하세요
print(df2.describe().loc[["mean", "max", "min"]])

