# Chapter 04 _ 4.1 벡터
# 벡터란, 유한한 차원의 공간에 존재하는 점들
# 벡터끼리 더하거나, 스칼라 상수와 곱해지면 새로운 벡터를 생성하는 '개념적 도구'
from typing import List
Vector = List[float]

height_weight_age = [70, 170, 40]
#                   인치, 파운드, 나이

grades = [95, 80, 75, 62]

# 벡터의 덧셈 정의
def add(v: Vector, w: Vector) -> Vector:
    """각 성분끼리 더한다"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i, in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

# 벡터의 뺄셈
def subtract(v: Vector, w: Vector) -> Vector:
    """각 성분끼리 뺀다"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]

# 벡터로 구성된 리스트에서 모든 벡터의 각 성분을 더하는 함수
def vector_sum(vectors: List[Vector]) -> Vector:
    """모든 벡터의 각 성분들끼리 더한다"""
    # vectors가 비어있는지 확인
    assert vectors, "no vectors provided!"

    # 모든 벡터의 길이가 동일한지 확인
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # i번째 결괏값은 모든 벡터의 i번째 성분을 더한 값
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

assert vector_sum([1, 2], [3, 4], [5, 6], [7, 8]) == [16, 20]

# 벡터에 스칼라 곱하기
def scalar_multiply(c: float, v: Vector) -> Vector:
    """모든 성분을 c로 곱하기"""
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

# 벡터로 구성된 리스트의 각 성분별 평균 구하기
def vector_mean(vectors: List[Vector]) -> Vector:
    """각 성분별 평균을 계산"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

# 벡터의 내적
# 내적이란, 벡터의 각 성분별 곱한 값을 더해준 것
def dot(v: Vector, w: Vector) -> float:
    """v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32    # 1 * 4 + 2 * 5 + 3 * 6

# 내적의 개념을 사용하여 각 성분으 제곱 값의 합 구하기
def sum_of_squares(v: Vector) - > float:
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

assert sum_of_squares([1, 2, 3]) == 14      # 1 * 1 + 2 * 2 + 3 * 3

# 제곱 값의 합을 이용하여 벡터의 크기 구하기
import math

def magnitude(v: Vector) -> float:
    """벡터 v의 크기를 반환"""
    return math.sqrt(sum_of_squares(v))     # math.sqrt()는 루트 계산

assert magnitude([3, 4]) == 5

# 두 벡터 간의 거리
def squared_distance(v: Vector, w: Vector) -> float:
    """(v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """벡터 v와 w간의 거리 계산"""
    return math.sqrt(squared_distance(v, w))

# 더욱 깔끔한 코드
def distance(v: Vector, w: Vector -> float:
    return magnitude(subtract(v, w))

             
# Chapter 04 _ 4.2 행렬
# 행렬은 2차원으로 구성된 숫자의 집합
# 리스트의 리스트로 표현 가능
# 리스트 안의 리스트들은 행(row)를 나타내며 모두 같은 길이를 가짐

# 타입 명시를 위한 별칭
Matrix = List[List[float]]

A = [[1, 2, 3],
     [4, 5, 6]]     # A는 2행 3열

B = [[1, 2],
     [3, 4],
     [5, 6]]        # B는 3행 2열

from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
    """(열의 개수, 행의 개수)를 반환"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0    # 첫 번째 행의 원소의 개수
    return num_rows, num_cols
assert shape([1, 2, 3], [4, 5, 6]) == (2, 3)       # 2행 3열

def get_row(A: Matrix, i = int) -> Vector:
    """A의 i번째 행을 반환"""
    return A[i]

def get_column(A: Matrix, j: int) -> Vector:
    """A의 j번째 열을 반환"""
    return [A_i[j],
            for A_i in A]

# 형태에 맞는 행렬 생성 후 각 원소를 채워주는 함수
from typing import Callable

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    (i, j)번째 원소가 entry_fn(i, j)인
    num_rows x num_cols 리스트를 반환
    """
    return [[entry_fn(i, j)
            for j in range(num_cols)]
            for i in range(num_rows)]

# 단위행렬 함수
def identify_matrix(n: int) -> Matrix:
    """n x n 단위 행렬을 반환"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identify_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]
