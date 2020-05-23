# Chapter 05 _ 5.1 데이터셋 설명하기
num_friends = [100, 49, 41, 40, 25, 30, 100, 49, 41, 40, 25, 30, 100, 49, 41, 40, 25, 30]

from collections import Counter
import matplotlib.pyplot as plt

friend_counts = Counter(num_friends)
xs = range(101)                     # 최댓값은 100
ys = [friend_counts[x] for x in xs]  # 히스토그램의 높이는 해당 친구 수를 갖고 있는 사용자 수
plt.bar(xs, ys)
plt.axis([0, 101, 0, 25])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
plt.show()


num_points = len(num_friends)

# 최댓값 최솟값
largest_value = max(num_friends)
smallest_value = min(num_friends)

sorted_values = sorted(num_friends)
smallest_value = sorted_values[0]
second_smallest_value = sorted_values[1]
second_largest_value = sorted_values[-2]


# Chapter 05 _ 5.1.1 중심 경향성
# 평균(average)을 구하는 함수
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

mean(num_friends)

# 중앙값(median) 함수
# 밑줄 표시로 시작하는 함수는 프라이빗 함수를 의미
# median 함수를 사용하는 사람이 직접 호출하는 것이 아닌
# median 함수만 호출하도록 생성됨
def _median_even(xs: List[float]) -> float:
    """len(xs)가 짝수면 두 중앙값의 평균을 반환"""
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2  # e.g. length 4 => hi_midpoint 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2

def median(v: List[float]) -> float:
    """v의 중앙값을 계산"""
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

assert median([1, 10, 2, 9, 5]) == 5
assert median([1, 9, 2, 10]) == (2 + 9) / 2

print(median(num_friends))

# 평균은 중앙값보다 계산하기 간편하지만, 이상치(outlier)에 매우 민감함

# 4분위수 구하는 함수
def quantile(xs: List[float], p: float) -> float:
    """x의 p분위에 속하는 값을 반환"""
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]

assert quantile(num_friends, 0.10) == 1
assert quantile(num_friends, 0.25) == 3
assert quantile(num_friends, 0.75) == 9
assert quantile(num_friends, 0.90) == 13

# 최빈값(mode, 데이터에서 가장 자주 나오는 값)을 구하는 함수
def mode(x: List[float]) -> List[float]:
    """최빈값이 하나보다 많을 수도 있으니 결과를 리스트로 반환"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]

assert set(mode(num_friends)) == {1, 6}


# Chapter 05 _ 5.1.2 산포도
# 산포도(dispension)는 데이터가 얼마나 퍼져있는지를 나타냄
# 보통 0과 근접한 값이면 데이터가 거의 퍼져 있지 않다는 의미이고
# 매우 큰 값이면 매우 퍼져있다는 것을 의미하는 통계치임
# 예로, 가장 큰 값과 작은 값의 차이를 나타내는 범위는 산포도를 나타내는 가장 간단한 통계치

# 파이썬에서 "range"는 이미 다른 것을 의미하기 때문에, 다른 이름을 사용함
def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)

assert data_range(num_friends) == 99

# 분산(variance)은 산포도를 측정하는 약간 더 복잡한 개념
from scratch.linear_algebra import sum_of_squares

def de_mean(xs: List[float]) -> List[float]:
    """x의 모든 데이터 포인트에서 평균을 뺌(평균을 0으로 만들기 위해)"""
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
    """편차의 제곱의 (거의) 평균"""
    assert len(xs) >= 2, "varience requires at leat two elements"

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)

assert 81.54 < varience(num_friends) < 81.55

# 표준편차(standard deviation)
import math

def standard_deviation(xs: List[float]) -> float:
    """표준편차는 분산의 제곱근"""
    return math.sqrt(variance(xs))

assert 9.02 < standard_deviation(num_friends) < 9.04

# 상위 25%에 해당되는 값과 하위 25%에 해당되는 값의 차이를 반환하는 함수
def interquartile_range(xs: List[float]) -> float:
    """상위 25%에 해당되는 값과 하위 25%에 해당되는 값의 차이를 반환"""
    return quantile(xs, 0.75) - quantile(xs, 0.25)

assert interquartile_range(num_friends) == 6


# Chapter 05 _ 5.2 상관관계
from scratch.linear_algebra import dot

def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs and ys must have same number of elements"

    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)

assert 22.42 < covariance(num_friends, daily_minutes) < 22.43
assert 22.42 / 60 < covariance(num_friends, daily_hours) < 22.43 / 60

# 상관관계
def correlation(xs: List[float], ys: List[float]) -> float:
    """xs와 ys의 값이 각각의 평균에서 얼마나 멀리 떨어져 있는지 계산"""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0        # 편차가 존재하지 않는다면 상관관계는 0

assert 0.24 < correlation(num_friends, daily_minutes) < 0.25
assert 0.24 < correlation(num_friends, daily_hours) < 0.25

# 이상치 제거
outlier = num_friends.index(100)
num_friends_good = [x for i, x in enumerate(num_friends)
                    if i != outlier]

daily_minutes_good = [x for i, x in enumerate(daily_minutes)
                      if i != outlier]

daily_hours_good = [dm / 60 for dm in daily_minutes_good]

assert 0.57 < correlation(num_friends_good, daily_minutes_good) < 0.58
assert 0.57 < correlation(num_friends_good, daily_hours_good) < 0.58
