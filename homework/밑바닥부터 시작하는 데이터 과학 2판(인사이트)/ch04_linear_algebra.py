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

