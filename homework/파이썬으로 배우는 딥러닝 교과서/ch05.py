# Chapter 05. 파이썬 기본 문법
# 5.1 리스트
# 5.1.1 리스트형(1)
'''문제
변수 c에 'red', 'blue', 'yellow' 세 개의 문자열을 저장
변수 c의 자료형을 print() 함수를 이용하여 출력
'''
c = ['red', 'blue', 'yellow']
print(type(c))                          # <class 'list'>


# 5.1.2 리스트형(2)
# [리스트 5-3] 리스트형(2)의 예
n = 3
print(['사과', n, '고릴라'])            # ['사과', 3, '고릴라']
'''문제
리스트형 fruits 변수를 만들어 apple, grape, banana 변수를 요소로 저장
'''
apple = 4
grape = 3
banana = 6
fruits = [apple, grape, banana]
print(fruits)                           # [4, 3, 6]


# 5.1.3 리스트 안의 리스트
# [리스트 5-6] 리스트 안의 리스트의 예
print([[1, 2], [3, 4], [5, 6]])         # [[1, 2], [3, 4], [5, 6]]
'''문제
변수 fruits는 '과일 이름'과 '개수' 변수를 가진 리스트
[['사과', 2], ['귤', 10]]이 출력되도록 fruits에 변수를 리스트형으로 대입
'''
fruits_name_1 = '사과'
fruits_num_1 = 2
fruits_name_2 = '귤'
fruits_num_2 = 10
fruits = [[fruits_name_1, fruits_num_1], [fruits_name_2, fruits_num_2]]
print(fruits)                           # [['사과', 2], ['귤', 10]]


# 5.1.4 리스트에서 값 추출
# [리스트 5-9] 리스트 값 꺼내기
a = [1, 2, 3, 4]
print(a[1])                             # 2
print(a[-2])                            # 3
'''문제
변수 fruits의 두 번째 요소를 출력하세요
변수 fruits의 마지막 요소를 출력하세요
'''
fruits = ['apple', 2, 'orange', 4, 'grape', 3, 'banana', 1]
print(fruits[1])                        # 2
print(fruits[-1])                       # 1


# 5.1.5 리스트에서 리스트 추출(슬라이스)
# [리스트 5-12] 슬라이스의 예
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
print(alphabet[1:5])                    # ['b', 'c', 'd', 'e']
print(alphabet[1:-5])                   # ['b', 'c', 'd', 'e']
print(alphabet[:5])                     # ['a', 'b', 'c', 'd', 'e']
print(alphabet[6:])                     # ['g', 'h', 'i', 'j']
print(alphabet[0:20])                   # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
'''문제
chaos 리스트에서 다음 리스트를 꺼내 변수 fruits에 저장
['apple', 2, 'orange', 4, 'grape', 3, 'banana', 1, 'elephant', 'dog']
변수 fruits를 print() 함수로 출력하세요
'''
chaos = ['cat', 'apple', 2, 'orange', 4, 'grape', 3, 'banana', 1, 'elephant', 'dog']
fruits = chaos[1:9]
print(fruits)                           # ['apple', 2, 'orange', 4, 'grape', 3, 'banana', 1]


# 5.1.6 리스트 요소 갱신 및 추가
# [리스트 5-15] 리스트 요소 갱신 및 추가
alphabet = ['a', 'b', 'c', 'd', 'e']
alphabet[0] = 'A'
alphabet[1:3] = ["B", "C"]
print(alphabet)                         # ['A', 'B', 'C', 'd', 'e']

alphabet = alphabet + ['f']
alphabet += ['g', 'h']
alphabet.append('i')
print(alphabet)                         # ['A', 'B', 'C', 'd', 'e', 'f', 'g', 'h', 'i']
'''문제
리스트 c의 첫 요소를 'red'로 갱신하세요
리스트 끝에 문자열 'green'을 추가하세요
'''
c = ['dog', 'blue',' yellow']
c[0] = 'red'
print(c)                                # ['red', 'blue', ' yellow']

c.append('green')
print(c)                                # ['red', 'blue', ' yellow', 'green']


# 5.1.7 리스트 요소 삭제
# [리스트 5-18] 리스트에서 요소를 삭제하는 방법
alphabet = ['a', 'b', 'c', 'd', 'e']
del alphabet[3:]
del alphabet[0]
print(alphabet)                         # ['b', 'c']
'''문제
변수 c의 첫 번째 요소를 제거하세요
'''
c = ['dog', 'blue', 'yello']
del c[0]
print(c)                                # ['blue', 'yello']


# 5.1.8 리스트형의 주의점
# [리스트 5-21] 리스트형의 예(1)
alphabet = ['a', 'b', 'c']
alphabet_copy = alphabet
alphabet_copy[0] = "A"
print(alphabet)                         # ['A', 'b', 'c']

# [리스트 5-22] 리스트형의 예(2)
alphabet = ['a', 'b', 'c']
alphabet_copy = alphabet[:]
alphabet_copy = "A"
print(alphabet)                         # ['a', 'b', 'c']

'''문제
변수 c의 리스트 요소가 변하지 않도록 'c_copy = c' 부분을 수정하세요
'''
c = ['red', 'blue', 'yellow']
c_copy = c[:]
c_copy[1] = 'green'
print(c)                                # ['red', 'blue', 'yellow']


# 5.2 딕셔너리
# 5.2.1 딕셔너리형
# [리스트 5-25] 딕셔너리형의 예
dic = {'Japan': 'Tokyo', 'Korea': 'Seoul'}
print(dic)                              # {'Japan': 'Tokyo', 'Korea': 'Seoul'}
'''문제
변수 town에 다음 키와 값을 가진 딕셔너리를 만들어 저장하세요
키1: 경기도, 값1: 수원, 키2: 서울, 값2: 중구
'''
town = {'경기도': '수원', '서울': '중구'}
print(town)                             # {'경기도': '수원', '서울': '중구'}


# 5.2.2 딕셔너리 요소 추출
# [리스트 5-28] 딕셔너리 요소를 추출하는 예
dic = {'Japan': 'Tokyo', 'Korea': 'Seoul'}
print(dic['Japan'])                     # Tokyo
'''문제
딕셔너리 town의 값을 이용하여 '경기도의 중심 도시는 수원입니다'라고 출력하세요
딕셔너리 town의 값을 이용하여 '서울의 중심 도시는 중구입니다'라고 출력하세요
'''
print("경기도의 중심 도시는 " + town['경기도'] + "입니다.")
# 경기도의 중심 도시는 수원입니다.
print("서울의 중심 도시는 " + town['서울'] + "입니다.")
# 서울의 중심 도시는 중구입니다.


# 5.2.3 딕셔너리 갱신 및 추가
# [리스트 5-31] 딕셔너리 갱신 및 추가의 예
dic = {'Japan': 'Tokyo', 'Korea': 'Seoul'}
dic['Japan'] = 'Osaka'
dic['China'] = 'Beijing'
print(dic)                  # {'Japan': 'Osaka', 'Korea': 'Seoul', 'China': 'Beijing'}

'''문제
키 '제주도'의 값 '제주시' 요소를 추가한 뒤 출력하세요
키 '경기도'의 값을 '분당'으로 변경하여 출력하세요
'''
town['경기도'] = '분당'
town['제주도'] = '제주시'
print(town)             # {'경기도': '분당', '서울': '중구', '제주도': '제주시'}


# 5.2.4 딕셔너리 요소 삭제
# [리스트 5-24] 딕셔너리 요소 삭제의 예
dic = {'Japan': 'Osaka', 'Korea': 'Seoul', 'China': 'Beijing'}
del dic['China']
print(dic)              # {'Japan': 'Osaka', 'Korea': 'Seoul'}
'''문제
키가 '경기도'인 요소를 삭제하세요
'''
del town['경기도']
print(town)             # {'서울': '중구', '제주도': '제주시'}


# 5.3 while문
# 5.3.1 while문(1)
# [리스트 5-37] while문의 예
# 'while 조건식: ...'
# 조건문이 True인 동안 whilw loop 실행
n = 2
while n > 0:
    print(n)
    n -= 1

'''문제
'''
x = 5
while x > 0:
    print('Hanbit')
    x -= 2                  # 3회 출력


# 5.3.2 while문(2)
'''문제
while문을 사용하여 변수 x가 0이 아닌 동안 반복하도록 만드세요
반복문 안에서는 변수 x에서 1을 빼고, x값을 출력하세요
다음의 실행 경과가 나오도록 만드세요
4
3
2
1
0
'''
x = 5
while x != 0:
    x -= 1
    print(x)


# 5.3.3 while문과 if문
'''문제
[리스트 5-41]에서 작성한 코드를 if문을 사용하여 다음과 같이 출력되도록 수정하세요
4
3
2
1
Bang
'''
x = 5
while x != 0:
    x -= 1
    if x != 0:
        print(x)
    else:
        print("Bang")


# 5.4 for문
# 5.4.1 for문
# [리스트 5-44] for문의 예
animals = ['tiger', 'dog', 'elephant']
for animal in animals:
    print(animal)
'''문제
for문을 사용하여 변수 storages의 요소를 하나씩 출력하세요
for문에서 사용할 변수명은 임의로 지정하세요
'''
storages = [1, 2, 3, 4]
for item in storages:
    print(item)


# 5.4.2 break
# [리스트 5-47] break의 예
storages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for item in storages:
    print(item)
    if n >= 5:
        print("끝")
        break

'''문제
변수 n의 값이 4일 때 처리를 종료하세요
'''
storages = [1, 2, 3, 4, 5, 6]
for n in storages:
    print(n)
    if n == 4:
        print("종료")
        break

# 5.4.3 continue
# [리스트 5-50] continue의 예
# continue는 break와 다르게 특정 조건일 때 루프를 한 번 건너뛴다.
storages = [1, 2, 3]
for n in storages:
    if n == 2:
        continue
    print(n)
'''문제
변수 n이 2의 배수일 때는 continue를 사용하여 처리를 생략하세요
'''
storages = [1, 2, 3, 4, 5, 6]
for n in storages:
    if n % 2 == 0:
        continue
    print(n)


# 5.5 추가설명
# 5.5.1 for문에서 index 표시
# for문에서 사용한 루프에서 리스트의 인덱스 확인이 필요할 때가 있다
# enumerate() 함수를 사용하면 인덱스가 포함된 요소를 얻을 수 있다
'''사용형식
for x, y in enumerate('리스트형'):
    for 안에서는 x, y를 사용하여 작성합니다
    x는 정수형의 인덱스, y는 리스트에 포함된 요소입니다
'''
# [리스트 5-53] for문에서 index를 표시하는 예
list = ['a', 'b']
for index, value in enumerate(list):
    print(index, value)
'''문제
for문 및 enumerate()를 사용하여 다음을 출력하는 코드를 작성하세요
index 0: tiger
index 1: dog
index 2: elephant
'''
animals = ['tiger', 'dog', 'elephant']
for index, animal in enumerate(animals):
    print("index " + str(index) + ":", animal)


# 5.5.2 리스트 안의 르스트 루프
# [리스트 5-56] 리스트 안의 리스트 루프의 예
list = [[1, 2, 3],
        [4, 5, 6]]
for a, b, c in list:
    print(a, b, c)
'''문제
for문을 사용하여 다음을 출력하는 코드를 작성
strawberry is red
peach is pink
banana is yellow
'''
fruits = [['strawberry', 'red'],
          ['peach', 'pink'],
          ['banana', 'yellow']]
for fruit, color in fruits:
    print(fruit + " is " + color)


# 5.5.3 딕셔너리형의 루프
# 딕셔너리형의 루프에서는 키와 값을 모두 변수로 하여 반복할 수 있다
# items()를 사용하여 'for key의 _변수명, value의_변수명 in 변수(딕셔너리형).iems():
# 로 기술한다
# [리스트 5-59] 딕셔너리형의 루프 예
fruits = {'strawberry': 'red', 'peach': 'pink', 'banana': 'yellow'}
for fruit, color in fruits.items():
    print(fruit + " is " + color)
'''문제
for문을 사용하여 다음을 출력하는 코드를 작성하세요
경기도 분당
서울 중구
제주도 제주시
'''
towns = {'경기도': '분당', '서울': '중구', '제주도': '제주시'}
for city, town in towns.items():
    print(city, town)


'''연습 문제
변수 items를 for문으로 루프시킵니다. 변수는 item으로 합니다
for문의 처리
 - '**는 한 개에 **원이며, **개 구입합니다'라고 출력
 - 변수 total_price에 가격 x 수량을 더해서 저장하세요
'지불해야 할 금액은 **원입니다'라고 출력하세요
변수 money에 임의의 값을 대입하세요
money > total_price일 때는 '거스름돈은 **원입니다'라고 출력하세요
money == total_price일 때는 '거스름돈은 없습니다'라고 출력하세요
money < total_price일 때는 '돈이 부족합니다'라고 출력하세요
'''
items = {'지우개': [100, 2], '펜': [200, 3], '노트': [400, 5]}
total_price = 0
for item in items:
    print(item + "은(는) 한 개에 " + str(items[item][0]) + "원이며, "
          + str(items[item][1]) + "개 구입합니다.")
    total_price += items[item][0] * items[item][1]
print("지불해야 할 금액은 " + str(total_price) + "원입니다.")
money = 3000
if money > total_price:
    print("거스름돈은 " + str(money - total_price) + "원입니다.")
elif money == total_price:
    print("거스름돈은 없습니다.")
else:
    print("돈이 부족합니다")
