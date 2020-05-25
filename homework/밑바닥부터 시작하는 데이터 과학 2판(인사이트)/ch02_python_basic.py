# Chapter 02 _ 2.4 들여쓰기
# '#' 기호는 주석을 의미한다
# 파이썬에서 주석은 실행되지 않지만, 코드를 이해하는데 도움이 된다.
for i in [1, 2, 3, 4, 5]:
    print(i)
    for j in [1, 2, 3, 4, 5]:
        print(j)
        print(i + j)
    print(i)
print("done looping")

long_winded_computation = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 +
                           13 + 14 + 15 + 16 + 17 + 18 + 19 + 20)

list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

easier_to_read_list_of_lists = [[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]

# 역슬래시(\)
two_plus_three = 2 + \
                 3

for i in [1, 2, 3, 4, 5]:

    # 빈 줄이 있다는 것을 확인하자.
    print(i)

# Chapter 02 _ 2.5 모듈
# import re
# my_regex = re.compile("[0-9]+", re.I)

# import re
# my_regex = regex.compile("[0-9]+", regex.I)

# import matplotlib.pyplot as plt
# plt.plot(...)

match = 10
from re import *
print(match)


# Chapter 02 _ 2.6 함수
def double(x):
    """
    이곳은 함수에 대한 설명을 적어 놓는 공간이다.
    예를 들어, '이 함수는 입력된 변수에 2를 곱한 값을 출력해준다'
    라는 설명을 추가할 수 있다.
    """
    return x * 2

def apply_to_one(f):
    """인자가 1인 함수 f를 호출"""
    return f(1)

my_double = double
x = apply_to_one(my_double)

# 람다 함수(lambda function)
y = apply_to_one(lambda x: x + 4)

# anouter_double = lambda x: 2 * x  이 방법은 최대한 피하자

def another_double(x):
    """대신 이렇게 작성하자"""
    return 2 * x

def my_print(message = "my default message"):
    print(message)

my_print("hello")
my_print()


def full_name(first = "What's-his-name", last = "Something"):
    return first + " " + last

full_name("Joel", "Grus")
full_name("Joel")
full_name(last = "Grus")


# Chapter 02 _ 2.7 문자열
single_quoted_string = 'data science'
double_quoted_string = "data science"

tab_string = "\t"
len(tab_string)

not_tab_string = r"\t"  # 문자 '\'와 't'를 나타내는 문자열
len(not_tab_string)

multi_line_string = """This is the first line.
and this is the second line
and this is the third line"""

first_name = "Joel"
last_name = "Grus"

full_name1 = first_name + " " + last_name       # 문자열 합치기
full_name2 = "{0} {1}".format(first_name, last_name)    # .format을 통한 문자열 합치기
full_name3 = f"{first_name} {last_name}"    # f-string ; 간편한 방법


# Chapter 02 _ 2.8 예외 처리, try와 except
try:
    print(0 / 0)
except ZeroDivisionError:
    print("cannot divide by zero")


# Chapter 02 _ 2.9 리스트
integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [integer_list, [heterogeneous_list], []]

list_length = len(integer_list)     # 결과는 3, 배열의 길이
list_sum = sum(integer_list)        # 결과는 6, 배열의 합

# 리스트 인덱싱 or 슬라이싱
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
zero = x[0]     # 결과는 0, 리스트의 순서는 0부터 시작한다.
one = x[1]      # 결과는 1
nine = x[-1]    # 결과는 9, 리스트의 마지막 항목을 가장 파이썬스럽게 불러오는 방법
eight = x[-2]   # 결과는 8, 뒤에서 두 번째 항목을 가장 파이썬스럽게 불러오는 방법
x[0] = -1       # x는 이제 [-1, 1, 2, 3, ..., 9]
                # 리스트 x의 첫번째(0번째) 항목의 값을 -1로 바꿈

first_three = x[:3]     # [-1, 1, 2]
three_to_end = x[3:]    # [3, 4, ..., 9]
one_to_four = x[1:5]    # [1, 2, 3, 4]
last_three = x[-3:]     # [7, 8, 9]
without_first_and_last = x[1:-1]    # [1, 2, ..., 8]
copy_of_x = x[:]        # [-1, 1, 2, ..., 9]

# 파이썬은 동일한 방식으로 리스트 뿐만 아니라 문자열 같은 순차형 변수를 나눌 수 있다.
# 또한 간격을 설정하여 리스트를 분리할 수도 있다.

every_third = x[::3]    # [-1, 3, 6, 9]
five_to_three = x[5:2:-1]   # [5, 4, 3]

# 파이썬에서 제공하는 in 연산자를 사용하면 리스트 안에서 항목의 존재 여부를 확인할 수 있다.
1 in [1, 2, 3]      # True
0 in [1, 2, 3]      # False

# 리스트 연결
x = [1, 2, 3]
x.extend([4, 5, 6])     # x는 이제 [1, 2, 3, 4, 5, 6]

# 만약 x를 수정하고 싶지 않다면 리스트를 더해 줄 수 도 있다.
x = [1, 2, 3]
y = x + [4, 5, 6]       # y는 이제 [1, 2, 3, 4, 5, 6], x는 변하지 않음

# 주로 리스트에 항목을 하나씩 추가하는 경우가 많다.
x = [1, 2, 3]
x.append(0)     # x는 이제 [1, 2, 3, 0]
y = x[-1]       # 결과는 0
z = len(x)      # 결과는 4

# 리스트 풀기, 버릴 항목은 밑줄(_)로 표시
x, y = [1, 2]       # x = 1, y = 2
_, y = [1, 2]       # y = 2, 밑줄은 신경쓰지 않음


# Chapter 02 _ 2.10 튜플
# 튜플은 변경, 수정할 수 없는 리스트
# 변경 수정할 수 없는 것을 제외하면 모든 기능은 리스트와 동일
# 튜플은 대괄호 대신 괄호를 사용하거나 기호 없이 적용
my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4
my_list[1] = 3      # my_list는 이제 [1, 3]

try:
    my_tuple[1] = 3
except TypeError:
    print("cannot modify a tuple")

# 함수에서 여러 값을 반환할 때 튜플을 사용할 수 있다.
def sum_and_product(x, y):
    return (x + y), (x * y)

sp = sum_and_product(2, 3)      # 결과는 (5, 6)
s, p = sum_and_product(5, 10)   # s는 15, p는 50

# 튜플과 리스트는 다중 할당을 지원한다.
x, y = 1, 2
x, y = y, x     # 가장 파이썬스럽게 변수를 교환; 이제 x는 2, y는 1


# Chapter 02 _ 2.11 딕셔너리
# 딕셔너리는 파이썬의 또 다른 기본적인 데이터 구조이며
# 특정 값과 연관된 키를 연결해 주고 이를 사용해 값을 빠르게 검색할 수 있음
empty_dict = {}     # 가장 파이썬스럽게 딕셔너리를 만드는 방법
empty_dict2 = dict()    # 덜 파이썬스럽게 딕셔너리를 만드는 방법
grades = {"Joel": 80,
          "Tim": 95}    # 딕셔너리 예시

# 대괄호를 사용하여 키의 값을 불러올 수 있다.
joels_grade = grades["Joel"]        # 결과는 80

try:
    Kates_grade = grades["Kate"]
except KeyError:
    print("no grade for Kate!")

# 연산자 in을 사용하면 키의 존재 여부를 확인할 수 있다.
joel_has_grade = "Joel" in grades   # True
kate_has_grade = "Kate" in grades   # False

# get() 메서드 활용
joels_grade = grades.get("Joel", 0)     # 결과는 80
kates_grade = grades.get("Kate", 0)     # 결과는 0
no_ones_grade = grades.get("No One")    # 기본적으로 None을 반환

# 대괄호를 사용해서 키와 값을 새로 지정해줄 수 있음
grades["Tim"] = 99
grades["Kate"] = 100
num_students = len(grades)

# 정형화된 데이터를 간단하게 나타낼 때에는 주로 딕셔너리가 사용됨
tweet = {"user": "joelgrus",
         "text": "Data Science is Awesome",
         "retweet_count": 100,
         "hashtags": ["#data", "#science", "#datascience", "#awesome", "#yolo"]}

# 특정 키 대신 딕셔너리의 모든 키를 한번에 살펴 볼 수 있다
tweet_keys = tweet.keys()       # 키에 대한 리스트
tweet_values = tweet.values()   # 값에 대한 리스트
tweet_items = tweet.items()     # (key, value) 튜플에 대한 리스트

"user" in tweet_keys
"user" in tweet
"joelgrus" in tweet_values


# Chapter 02 _ 2.11.1 defaultdict
word_counts = {}
for word in document:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

# 예외처리 딕셔너리 생성
word_counts = {}
for word in document:
    try:
        word_counts[word] += 1
    except KeyError:
        word_counts[word] = 1

# 존재하지 않는 키를 적절하게 처리해주는 get을 사용해서 딕셔너리 생성
word_counts = {}
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1

# defaultdict 사용
from collections import defaultdict

word_counts = defaultdict(int)      # int()는 0을 생성
for word in document:
    word_counts[word] += 1

# 리스트, 딕셔너리 혹은 직접 만든 함수를 인자에 넣어줄 수 있음
dd_list = defaultdict(list)     # list()는 빈 리스트를 생성
dd_list[2].append(1)            # 이제 dd_list는 {2: [1]}을 포함

dd_dict = defaultdict(dict)     # dict()는 빈 딕셔너리 생성
dd_dict["Joel"]["City"] = "Seattle"     # {"Joel : {"City": Seattle"}}

dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1           # 이제 dd_pair는 {2: [0, 1]}을 포함
