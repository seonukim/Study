# Chapter 07. NumPy
# 7.1.2 NumPy의 고속 처리 경험하기
'''문제'''
# 필요한 라이브러리를 import합니다.
import numpy as np
import time
from numpy.random import rand
'''
# 행, 열의 크기
N = 150

# 행렬을 초기화합니다
matA = np.array(rand(N, N))
matB = np.array(rand(N, N))
matC = np.array([[0] * N for _ in range(N)])

# 파이썬의 리스트를 사용하여 계산합니다
# 시작 시간을 지정합니다
start = time.time()

# for문을 사용하여 행렬 곱셈을 실행합니다
for i in range(N):
    for j in range(N):
        for k in range(N):
            matC[i][j] = matA[i][k] * matB[k][j]
print("파이썬 기능만으로 계산한 결과: %.2f[sec]" & float(time.time() - start))

# NumPy를 사용하여 계산합니다
# 시작 시간을 저장합니다
start = time.time()

# NumPy를 사용하여 행렬 곱셈을 실행합니다
matC = np.dot(matA, matB)

# 소수점 이하 두 자리까지 표시되므로 NumPy는 0.00[sec]로 표시됩니다
print("NumPy를 사용하여 계산한 결과: %.2f[sec]" % float(time.time() - start))
'''

# 7.2 NumPy 1차원 배열
# 7.2.1 import
'''문제
NumPy를 import하여 np라는 이름으로 정의하세요
'''
# import numpy as np


# 7.2.2 1차원 배열
'''문제
변수 storages에서 ndarray 배열을 생성하여 변수 np_storages에 대입하세요
변수 np_storages의 자료형을 출력하세요
'''
np_storages = np.array([24, 3, 4, 23, 10, 12])
print(type(np_storages))


# 7.2.3 1차원 배열의 계산
# [리스트 7-7] 1차원 배열 계산의 예
# NumPy를 사용하지 않고 실행합니다
storages = [1, 2, 3, 4]
new_storages = []
for n in storages:
    n += n
    new_storages.append(n)
print(new_storages)

# [리스트 7-8] 1차원 배열 계산의 예
# NumPy를 사용하여 실행합니다
import numpy as np
storages = np.array([1, 2, 3, 4])
storages += storages
print(storages)
'''문제
변수 arr에 arr을 더해서 출력
변수 arr에 arr을 빼서 출력
변수 arr의 3승을 출력
1을 변수 arr로 나눠서 출력
'''
arr = np.array([2, 5, 3, 4, 8])
# arr + arr
print('arr + arr : ', arr + arr)
print('arr - arr : ', arr - arr)
print('arr ** 3 : ', arr ** 3)
print('1 / arr : ', 1 / arr)


# 7.2.4 인덱스 참조와 슬라이스
# [리스트 7-11] 슬라이스의 예
arr = np.arange(10)
print(arr)

arr = np.arange(10)
arr[0:3] = 1
print(arr)
'''문제
변수 arr의 요소 중에서 3, 4, 5만 출력
변수 arr의 요소 중에서 3, 4, 5를 24로 변경
'''
arr = np.arange(10)
print(arr)
print(arr[3:6])
arr[3:6] = 24
print(arr)

'''문제
[리스트 7-15]의 코드를 실행하여 동작을 확인하세요
'''
# nparray를 그대로 arr2변수에 대입한 경우
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)

arr2 = arr1
arr2[0] = 100
# arr2 변수를 변경하면 원래 변수 arr1도 영향을 받는다
print(arr1)

# ndarray를 copy()를 사용해서 arr2 변수에 대입한 경우
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)

arr2 = arr1.copy()
arr2[0] = 100

# arr2 변수를 변경해도 원래 변수에 영향을 주지 않는다
print(arr1)


# 7.2.6 view와 copy
'''문제
[리스트 7-17]을 실행하여 파이썬의 리스트와 numpy의 ndarray 슬라이스의 차이를 확인하세요
'''
import numpy as np

# 파이썬의 리스트에 슬라이스를 이용한 경우를 확인합니다
arr_List = [x for x in range(10)]
print("리스트 형 데이터")
print("arr_List :", arr_List)

print()
arr_List_copy = arr_List[:]
arr_List_copy[0] = 100

print("리스트의 슬라이스는 복사본이 생성되므로, arr_List에는 arr_List_copy의 변경점이 반영되지 않는다.")
print("arr_List:",arr_List)
print()

# NumPy의 ndarray에 슬라이스를 이용한 경우를 확인합니다
arr_NumPy = np.arange(10)
print("NumPy의 ndarray 데이터")
print("arr_NumPy:",arr_NumPy)
print()

arr_NumPy_view = arr_NumPy[:]
arr_NumPy_view[0] = 100

print("NumPy의 슬라이스는 view(데이터가 저장된 위치의 정보)가 대입되므로, arr_NumPy_view를 변경하면 arr_NumPy에 반영됨")
print("arr_NumPy:",arr_NumPy)
print()

# NumPy의 ndarray에서 copy()를 이용한 경우를 확인합니다
arr_NumPy = np.arange(10)
print("NumPy의 ndarray에서 copy()를 이용한 경우")
print("arr_NumPy:",arr_NumPy)
print()
arr_NumPy_copy = arr_NumPy[:].copy()
arr_NumPy_copy[0] = 100

print("copy()를 사용하면 복사본이 생성되기 때문에 arr_NumPy_copy는 arr_NumPy에 영향을 미치지 않는다")
print("arr_NumPy:",arr_NumPy)


# 7.2.7 부울 인덱스 참조
# [리스트 7-19] 부울 인덱스 참조의 예
arr = np.array([2, 4, 6, 7])
print(arr[np.array([True, True, True, False])])     # [2 4 6]

# [리스트 7-20] 부울 인덱스 참조의 예
arr = np.array([2, 4, 6, 7])
print(arr[arr % 3 == 1])                            # [4 7]

'''문제
변수 arr의 각 요소가 2로 나누어 떨어지는지 나타내는 부울 배열 출력
변수 arr의 각 요소 중 2로 나누어떨어지는 요소의 배열 출력
'''
arr = np.array([2, 3, 4, 5, 6, 7])

print(arr % 2 == 0)         # [ True False  True False  True False]
print(arr[arr % 2 == 0])    # [2 4 6]


# 7.2.8 범용 함수
'''문제
변수 arr의 각 요소를 절댓값으로 하여 변수 arr_abs에 대입
변수 arr_abs의 각 요소의 e의 거듭제곱과 제곱근을 출력
'''
arr = np.array([4, -9, 16, -4, 20])
print(arr)                  # [ 4 -9 16 -4 20]

arr_abs = np.abs(arr)
print(arr_abs)              # [ 4  9 16  4 20]

print(np.exp(arr_abs))      # [5.45981500e+01 8.10308393e+03 8.88611052e+06 5.45981500e+01 4.85165195e+08]
print(np.sqrt(arr_abs))     # [2.         3.         4.         2.         4.47213595]


# 7.2.9 집합함수
'''문제
np.unique() 함수를 사용하여 변수 arr1에서 중복을 제거한 배열을 변수 new_arr1에 대입
변수 new_arr1과 변수 arr2의 합집합을 출력
변수 new_arr1과 변수 arr2의 교집합, 차집합 출력
'''
arr1 = [2, 5, 7, 9, 5, 2]
arr2 = [2, 5, 8, 3, 1]

# np.unique() 함수를 사용하여 중복을 제거한 배열을 변수 new_arr1에 대입
new_arr1 = np.unique(arr1)
print(new_arr1)

# 합집합
print(np.union1d(new_arr1, arr2))

# 교집합
print(np.intersect1d(new_arr1, arr2))

# 차집합
print(np.setdiff1d(new_arr1, arr2))


# 7.2.10 난수
'''문제
np.random을 적지 않아도 randint()함수를 사용할 수 있도록 import
변수 arr1에 각 요소가 0 이상 10 이하인 정수 행렬 (5 x 2)을 대입
변수 arr2에 0 이상 1 미만의 난수 3개 생성해서 대입
'''
# np.random을 적지 않아도 randint() 함수를 사용할 수 있도록 import
from numpy.random import randint

# 변수 arr1에 각 요소가 0 이상 10 이하인 정수 행렬(5 × 2)를 대입
arr1 = randint(0, 11, (5, 2))
print(arr1)

# 변수 arr2에 0 이상 1 미만의 난수를 3개 생성해서 대입
arr2 = np.random.rand(3)
print(arr2)


# 7.2 넘파이 2차원 배열
# 7.3.1 2차원 배열
'''문제
변수 arr에 2차원 배열을 대입
변수 arr 행렬의 각 차원의 요소 수를 출력
변수 arr을 4행 2열의 행렬로 변환합니다
'''
# 변수 arr에 2차원 배열을 대입
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr)

# 변수 arr 행렬의 각 차원의 요소 수를 출력
print(arr.shape)

# 변수 arr을 4행 2열의 행렬로 변환
print(arr.reshape(4, 2))


# 7.3.2 인덱스 참조와 슬라이스
# [리스트 7-31] 인덱스 참조의 예
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[1])           # [4 5 6]

# [리스트 7-32] 인덱스 참조의 예
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[1][2])            # 6
'''문제
변수 arr의 요소 중 3을 출력
변수 arr에서 부분적으로 꺼내어 출력
1행 이후, 2열까지를 꺼낸다
'''
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr) 

# 변수 arr의 요소 중 3을 출력
print(arr[0, 2])

# 변수 arr에서 부분적으로 꺼내어 출력
# "1행 이후, 2열까지"를 꺼낸다
print(arr[1:, :2])


# 7.3.3 axis
'''문제
arr 행의 합을 구하여 다음과 같은 1차원 배열을 반환하세요
'''
arr = np.array([[1, 2, 3], [4, 5, 12], [15, 20, 22]])
print(arr.sum(axis=1))          # [ 6 21 57]


# 7.3.4 팬시 인덱스 참조
# [리스트 7-39] 팬시 인덱스 참조의 예
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(arr[[3, 2, 0]])
'''
[[7 8]
 [5 6]
 [1 2]]
'''
'''문제
팬시 인덱스 참조를 사용하여 변수 arr의 2행, 4행 1행 순서로 배열을 출력
여기서 말하는 행은 인덱스 번호와는 별도로 1행부터 센 행이다
'''
arr = np.arange(25).reshape(5, 5)
print(arr[[1, 3, 0]])
'''
[[ 5  6  7  8  9]
 [15 16 17 18 19]
 [ 0  1  2  3  4]]
'''


# 7.3.5 전치 행렬
'''문제
변수 arr을 전치하여 출력하세요
'''
arr = np.arange(10).reshape(2, 5)

# 변수 arr을 전치
print(np.transpose(arr))
'''
[[0 5]
 [1 6]
 [2 7]
 [3 8]
 [4 9]]
'''


# 7.3.6 정렬
# [리스트 7-44] 정렬의 예
arr = np.array([15, 30, 5])
print(arr.argsort())
'''문제
변수 arr을 argsort() 메서드로 정렬하여 출력
변수 arr을 np.sort() 함수로 정렬하여 출력
변수 arr을 sort() 메서드로 행을 정렬하여 출력
'''
arr = np.array([[8, 4, 2], [3, 5, 1]])

# argsort() 메서드로 출력
print(arr.argsort())

# np.sort() 함수로 정렬하여 출력
print(np.sort(arr))

# sort() 메서드로 행을 정렬
arr.sort(1)
print(arr)


# 7.3.7 행렬 계산
'''문제
변수 arr과 arr의 행렬곱을 출력
변수 vec의 노름을 출력
'''
arr = np.arange(9).reshape(3, 3)
print(np.dot(arr, arr))

vec = arr.reshape(9)
print(np.linalg.norm(vec))


# 7.3.8 통계 함수
'''문제
변수 arr의 각 열의 평균을 출력
변수 arr의 행 합계를 출력
변수 arr의 최소값을 출력
변수 arr의 각 열의 최대값의 인덱스 번호를 출력
'''
arr = np.arange(15).reshape(3, 5)

# 변수 arr의 각 열의 평균을 출력
print(arr.mean(axis = 0))

# 변수 arr의 행 합계를 출력
print(arr.sum(axis = 1))

# 변수 arr의 최소값을 출력
print(arr.min())

# 변수 arr의 각 열의 최대값의 인덱스 번호를 출력
print(arr.argmax(axis = 0))


# 7.3.9 브로드캐스트
# [리스트 7-51] 브로드캐스트의 예
x = np.arange(6).reshape(2, 3)
print(x + 1)
'''
[[1 2 3]
 [4 5 6]]
'''
'''문제
0에서 14 사이의 정수값을 가진 ndarray 배열 x에서 0에서 4 사이의 정수값을 가진
ndarray 배열 y를 빼세요
'''
# 0에서 14 사이의 정수값을 갖는 3 × 5의 ndarray 배열 x를 생성
x = np.arange(15).reshape(3, 5)
print("x : ", x)

# 0에서 4 사이의 정수값을 갖는 1 × 5의 ndarray 배열 y를 생성
y = np.array([np.arange(5)])
print("y : ", y)

# x의 n번째 열의 모든 행에서 n만 뺌
z = x - y

# x를 출력
print("z : ", z)

'''연습 문제'''
'''문제
각 요소가 0 ~ 30인 정수 행렬(5 × 3)을 변수 arr에 대입하세요
변수 arr을 전치하세요
변수 arr의 2, 3, 4열만 추출한 행렬(3 × 3)을 변수 arr1에 대입하세요
변수 arr1의 행을 정렬하세요
각 열의 평균을 출력하세요
'''
np.random.seed(100)

# 각 요소가 0 ~ 30인 정수 행렬(5 × 3)을 변수 arr에 대입하세요
arr = np.random.randint(0, 31, (5, 3))
print(arr)

# 변수 arr을 전치하세요
arr = arr.T
print(arr)

# 변수 arr의 2, 3, 4열만 추출한 행렬(3 × 3)을 변수 arr1에 대입하세요
arr1 = arr[:, 1:4]
print(arr1)

# 변수 arr1의 행을 정렬하세요
arr1.sort(0)
print(arr1)

# 각 열의 평균을 출력하세요
print(arr1.mean(axis = 0))


'''종합 문제'''
'''문제
난수로 지정한 크기의 이미지를 생성하는 함수 make_image()를 완성
전달된 행렬의 일부분을 난수로 변경하는 함수 change_matrix()를 완성
생성된 image1과 image2의 각 요소의 차이의 절댓값을 계산하여 image3에 대입
'''
np.random.seed(0)

def make_image(m, n) :
    image = np.random.randint(0, 6, (m, n)) 
    return image

def change_little(matrix) :
    shape = matrix.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.randint(0, 2)==1:
                matrix[i][j] = np.random.randint(0, 6, 1)
    return matrix

image1 = make_image(3, 3)
print(image1)
print()

image2 = change_little(np.copy(image1))
print(image2)
print()

image3 = image2 - image1
print(image3)
print()

image3 = np.abs(image3)
print(image3)
