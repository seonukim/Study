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