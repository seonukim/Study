# 람다(lambda) : 익명 함수
pow2 = lambda x: x ** 2

# 람다식의 구조
# lambda 인수: 반환 값

'''문제
인수 a에 대한 다음 함수를 작성하고, a = 4일 때의 반환값을 각각 출력하세요
def를 이용하여 2a^2 - 3a + 1을 출력하는 함수 func1
lambda를 이용하여 2a^2 - 3a + 1을 출력하는 함수 func2
'''
# def로 정의
def func1(a):
    return print(2 * a ** 2 - 3 * a + 1)
func1(3)

# lambda로 정의
func2 = lambda a: 2 * a ** 2 - 3 * a + 1
print(func2(3))

def func3(x, y, z):
    return x * y + z
print(func3(3, 6, 9))

func4 = lambda x, y, z: x * y + z
print(func4(1, 2, 3))

# if를 이용한 람다
lower_three2 = lambda x: x * 2 if x < 3 else x // 3 + 5
print(lower_three2(3))

# 삼항 연산자의 표기
# 조건을 만족할 경우 if 조건 else 조건을 만족하지 않을 경우

# a가 10 이상 30 미만이면 a^2 - 40a + 350의 계산값을,
# 그 이외의 경우에는 50을 반환하는 함수 func5
a1 = 13
a2 = 32
func5 = lambda x: x**2 - 40*x + 350 if 10 <= x < 30 else 50
print(func5(a1))
print(func5(a2))

test_sentence = "this is a test sentence."
test_sentence = test_sentence.split(" ")
print(test_sentence)

self_data = "My name is Seonu"
self_data = self_data.split(' ')
print(self_data[3])

import re
test_sentence = "this,is a.test,sentence"
test_sentence = re.split("[, .]", test_sentence)
print(test_sentence)
# re.split("[구분_기호]", 분할할 문자열)

# "년/월/일_시:분" 구조를 갖는 문자열 time_data를 분할하여 "월"과 "시"를 꺼내 출력하세요
time_data = "2020년/6월/29일_11시:41분"
# print(time_data)
time_data = re.split("[/_:]", time_data)
print(time_data[1])
print(time_data[3])

time_list = [
    "2006/11/26_2:40",
    "2009/1/16_23:35",
    "2014/5/4_14:26",
    "2017/8/9_7:5",
    "2020/1/5_22:15"
]

get_hour = lambda x: int(re.split("[/_:]", x)[3])
hour_list = list(map(get_hour, time_list))
print(hour_list)

a = [1, -2, 3, -4, 5]
new = []
for x in a:
    if x > 0:
        new.append(x)
print(new)

print(list(filter(lambda x: x > 0, a)))
# 반복자
# filter(조건이 되는 함수, 배열)

# 리스트에 저장
# list(filter(함수, 배열))

# time_list에서 "월"이 1 이상 6 이하인 요소를 추출하여 배열로 출력
get_month = lambda x: int(re.split("[/_:]", x)[1]) - 7 < 0
print(list(filter(get_month, time_list)))

# sorted
nest_list = [
    [0, 9],
    [1, 8],
    [2, 7],
    [3, 6],
    [4, 5]
]

print(sorted(nest_list, key = lambda x: x[1]))
# 키를 설정해서 정렬한다
# sorted(정렬하려는 배열, key = 키가 되는 함수, reverse = True or False(default = False))

# "시"가 오름차순이 되도록 time_data를 정렬해서 출력하세요
time_data = [
    [2006, 11, 26, 2, 40],
    [2009, 1, 16, 23, 35],
    [2014, 5, 4, 14, 26],
    [2017, 8, 9, 7, 5],
    [2020, 1, 5, 22, 15]
]

print(sorted(time_data, key = lambda x: x[3]))

# 리스트 내포(list comprehension)
a = [1, -2, 3, -4, 5]
print([abs(x) for x in a])
# [적용하려는 함수 for 요소 in 적용할 원본 배열]

# 측정된 시간(분)을 저장한 minute_data에서 해당 시간을 [시, 분]으로 변환한 배열을 만들어 출력하세요
# 75분을 60으로 나누면 몫은 1, 나머지는 15
# 따라서 75분은 1시간 15분이다
minute_data = [30, 155, 180, 74, 11, 60, 82]
h_m_split = lambda x: [x // 60, x % 60]
h_m_data = [h_m_split(x) for x in minute_data]
print("=" * 40)
print(h_m_data)

# if문을 이용한 루프
a = [1, -2, 3, -4, 5]
print("=" * 40)
print([x for x in a if x > 0])

# 후위(postfix notation) if의 사용법은 다음과 같다
# [적용할 함수(요소) for 요소 in 필터링할 배열 if 조건]
# minute_data를 [시, 분]으로 변환하여 [시, 0]인 요소만 추출하라
just_hour_data = [x for x in minute_data if x % 60 == 0]
print(just_hour_data)

# 여러 배열을 동시에 루프시키기
# zip() 함수 이용하기
a = [1, -2, 3, -4, 5]
b = [9, 8, -7, -6, -5]
for x, y in zip(a, b):
    print(x, y)

# 리스트 컴프리헨션에서도 마찬가지로 zip() 함수를 사용할 수 있다
print([x**2 + y**2 for x, y in zip(a, b)])

# hour와 minute 배열을 분으로 변환하고, 배열로 만들어 출력하세요
# 분으로 변환하려면 '시간 * 60 + 분'
hour = [0, 2, 3, 1, 0, 1, 1]
minute = [30, 35, 0, 14, 11, 0, 22]
h_m_combined = lambda x, y: x*60 + y
minute_data1 = [h_m_combined(x, y) for x, y in zip(hour, minute)]
print(minute_data1)

# 다중루프
# 루프 속의 루프
a = [1, -2, 3]
b = [9, 8]

for x in a:
    for y in b:
        print(x, y)

# 리스트 컴프리헨션 -> for를 나란히 두 번 사용하기
print([[x, y] for x in a for y in b])

# 이진수를 십진수로 변환하기
# 십진수 : (셋째 자리 숫자) * 10^2 + (둘째 자리 숫자) * 10 + (첫째 자리 숫자)
# 이진수 : (셋째 자리 숫자) * 2^2 + (둘째 자리 숫자) * 2 + (첫째 자리 숫자)
threes_place = [0, 1]
twos_place = [0, 1]
ones_place = [0, 1]
for x in threes_place:
    for y in twos_place:
        for z in ones_place:
            print(x*2**2 + y*2 + z)

digit = [x*4 + y*2 + z for x in threes_place for y in twos_place for z in ones_place]
print(digit)

# 딕셔너리 객체
# defaultdict
# 딕셔너리의 요소가 출현한 횟수를 기록합니다
d = {}
lst = ['foo', 'bar', 'pop', 'pop', 'foo', 'popo']
for key in lst:
    # d에 key가 존재하느냐에 따라 분류합니다
    if key in d:
        d[key] += 1
    else:
        d[key] = 1
print(d)

from collections import defaultdict
d = defaultdict(int)
for key in lst:
    d[key] += 1
print(d)

