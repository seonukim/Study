# 6.1 내장 함수와 메서드
# 6.1.1 내장 함수
'''문제
변수 vege의 객체 길이를 len() 함수와 print() 함수를 이용하여 출력하세요
변수 n의 객체 길이를 len() 함수와 print() 함수를 이용하여 출력하세요
'''
vege = 'potato'
n = [4, 5, 2, 7, 6]
print(len(vege))
print(len(n))


# 6.1.2 메서드
'''문제
다음 코드를 실행했을 때의 출력 결과를 선택하세요
'''
# 첫 번째
alphabet = ['b', 'a', 'e', 'c', 'd']
sorted(alphabet)
print(alphabet)
print(sorted(alphabet))

# 두 번째
alphabet = ['b', 'a', 'e', 'c', 'd']
alphabet.sort()
print(alphabet)


# 6.1.3 문자열형 메서드(upper, count)
# [리스트 6-6] upper() 메서드와 count() 메서드의 예
city = 'Tokyo'
print(city.upper())
print(city.count('o'))
'''문제
변수 animal에 저장되어 있는 문자열을 대문자로 변환해서 변수 animal_big에 저장
변수 animal에 'e'가 몇 개 포함되어 있는지 출력
'''
animal = 'elephant'
animal_big = animal.upper()
print(animal_big)                   # ELEPHANT
print(animal.count('e'))            # 2


# 6.1.4 문자열형 메서드 format
'''문제
format() 메서드를 사용하여 '바나나는 노란색입니다'를 출력
'''
fruit = '바나나'
color = '노란색'
print("{}는 {}입니다.".format(fruit, color))


# 6.1.5 리스트형 메서드 index
'''문제
'2'의 인덱스 번호를 출력하세요
변수 n에 '6'이 몇 개 들어 있는지 출력하세요
'''
n = [3, 6, 8, 6, 3, 2, 4, 6]
print(n.index(2))       # 5
print(n.count(6))       # 3


# 6.1.6 리스트형 메서드 sort
'''문제
변수 n을 정렬하여 오름차순으로 출력
n.reverse()로 정렬된 변수 n의 요소의 순서를 반대로 하여 내림차순으로 출력
'''
n = [53, 26, 37, 69, 24, 2]
n.sort()
print(n)

n.reverse()
print(n)


# 6.2 함수
# 6.2.1 함수 작성
# [리스트 6-19] 함수 작성의 예
def sing():
    print("노래합니다!")

'''문제
'홍길동입니다'라고 출력하는 함수 introduce를 작성하세요
'''
def introduce():
    print("홍길동입니다.")
introduce()


# 6.2.2 인수
# [리스트 6-22] 인수의 예
def introduce(n):
    print(n + "입니다.")
introduce("홍길동")

'''문제
인수 n을 세 제곱한 값을 출력하는 함수 cube_cal을 작성하세요
'''
def cube_cal(n):
    print(n ** 3)

cube_cal(4)


# 6.2.3 복수 개의 인수
'''문제
인수 n과 age를 이용하여 '**입니다. **살입니다.'를 출력하는 함수 introduce를 작성
홍길동과 18을 인수로 하여 introduce를 호출
'''
def introduce(n, age):
    print(n + "입니다." + str(age) + "살입니다.")
introduce('홍길동', 18)


# 6.2.4 인수의 초깃값
'''문제
인수 n의 초깃값을 '홍길동'으로 합니다.
'18'만 인수로 넣어 함수를 호출하세요
'''
def introduce(age, n = '홍길동'):
    print(n + '입니다.' + str(age) + "살입니다.")
introduce(18)


# 6.2.5 return
# [리스트 6-33] return의 예(1)
def introduce(first = '김', second = '길동'):
    return '성은 ' + first + "이고, 이름은 " + second + "입니다."
print(introduce('홍'))

# [리스트 6-34] return의 예(2)
# def introduce(first = '김', second = '길동'):
#     comment = '성은 ' + first + "이고, 이름은 " + second + "입니다.")
#     return comment
# print(introduce('홍'))

'''문제
bmi를 계산하는 함수를 작성하고, bmi 값을 반환하세요
bmi = weight / height ** 2로 계산할 수 있습니다
weight, height라는 두 변수를 사용하세요
'''
def bmi(weight, height):
    return weight / height ** 2
print(bmi(175, 73))


# 6.2.6 함수 임포트(가져오기)
# 유사한 용도끼리 셋으로 묶여서 제공되는 것 = 패키지
# 패키지 안에 들어 있는 하나하나의 함수 = 모듈
import time
now_time = time.time()
print(now_time)

from time import time
now_time = time()
print(now_time)
'''문제
from을 이용하여 time 패키지의 time 모듈을 import하세요
time()을 이용하여 현재 시간을 출력하세요
'''
from time import time
now_time = time()
print(now_time)


# 6.3.2 클래스(멤버와 생성자)
# [리스트 6-42] 클래스의 예
class MyProduct:
    def __init__(self, name, price):
        self.name = name
        self.price = price
        self.stock = 0
        self.sales = 0
# [리스트 6-43] 클래스의 예
    # product1 = MyProduct('cake', 500)
'''문제
MyProduct 클래스의 생성자를 수정하여 클래스 호출 시 name, product, stock의
초깃값을 설정하세요. 이때 각각의 인수명은 다음과 같습니다.
 - 상품명 : name
 - 가격  : price
 - 재고  : stock
 '''
class MyProduct:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0
    
    product_1 = MyProduct('cake', 500, 20)
    print(product_1.stock)


# 6.3.3 클래스(메서드)
# [리스트 6-46] 클래스(메서드)의 예
class MyProduct:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0
    
    # 구매 메서드
    def buy_up(self, n):
        self.stock += n
    
    # 판매 메서드
    def sell(self, n):
        self.stock -= n
        self.sales += n * self.price
    
    # 개요 메서드
    def summary(self):
        message = "called summary(). \n name: " + self.name + \
        "\n price: " + str(self.price) + \
        "\n stock: " + str(self.stock) + \
        "\n sales: " + str(self.sales)
        print(message)

'''연습 문제 : 순차 검색 알고리즘'''
def binary_search(numbers, target_number):
    # 최솟값 임시 결정
    low = 0
    # 범위 내의 최댓값
    high = len(numbers)
    while low <= high:
        # 중앙값 구하기(index)
        middle = (low + high) // 2
        # numbers(검색 대상)의 중앙값과 target_number(찾는 값)가 동일한 경우
        if numbers[middle] == target_number:
            # 출력
            print("{1}은(는) {0}번째에 있습니다.".format(middle, target_number))
            # 종료
            break
        # numbers의 중앙값이 target_number보다 작은 경우
        elif numbers[middle] < target_number:
            low = middle + 1
        # numbers의 중앙값이 target_number보다 큰 경우
        else:
            high = middle - 1
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
target_number = 11
binary_search(numbers, target_number)