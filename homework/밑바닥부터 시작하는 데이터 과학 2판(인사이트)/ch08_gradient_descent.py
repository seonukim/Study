# Chapter 08 _ 8.1 경사하강법에 숨은 의미
# 실수 벡터를 입력하면 실수 하나를 출력해 주는 함수 f
from scratch.linear_algebra import Vector, dot

def sum_of_squares(v: Vector) -> float:
    """v에 속해있는 항목들의 제곱합을 계산"""
    return dot(v, v)

# 그래디언트(gradient, 경사, 기울기; 이것은 편미분 벡터)
# 함수가 가장 빠르게 증가할 수 있는 방향을 나타낸다


# Chapter 08 _ 8.2 그래디언트 계산하기
from typing import Callable
def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h

# 도함수 구하기
def square(x: float) -> float:
    return x * x

def derivative(x: float) -> float:
    return 2 * x

# 도함수를 구할 수 없다면, 아주 작은 e값을 대입해 미분값을 어림 잡음
xs = range(- 10, 11)
acuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h = 0.001) for x in xs]

# 두계산식의 결과값이 거의 비슷함을 보여주기 위한 그래프
import matplotlib.pyplt as plt
plt.title("Actual Derivates vs. Estimates")
plt.plot(xs, actuals, 'rx', label = 'Actual')       # 빨간색 x
plt.plot(xs, estimates, 'b+', label = 'Estimate')   # 파란색 +
plt.legend(loc = 9)
plt.show()

# 편도함수
# i번째 편도함수는 i번째 변수를 제외한 다른 모든 입력 변수를 고정시켜 계산
def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """함수 f의 i번째 편도함수가 v에서 가지는 값"""
    w = [v_j + (h if j == i else 0)    # h를 v의 i번째 변수에만 더해 준다
        for j, v_j in enumerate(v)]
    
    return (f(w) - f(v)) / h

# 일반적인 도함수와 같은 방법으로 그래디언트의 근사값을 구한다
def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]


# Chapter 08 _ 8.3 그래디언트 적용하기
# 경사하강법을 이용하여 3차원 벡터의 최솟값 구하기
# 임의의 시작점을 잡고, 그래디언트가 아주 작아질 때까지 경사의 반대 방향으로 조금씩 이동
import random
from scratch.linear_algebra import distance, add, scalar_multiply

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """v에서 step_size만큼 이동하기"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

# 임의의 시작점 선택
v = [random.uniform(-10, 10) for i in range(3)]
    
for epoch in range(1000):
    grad = sum_of_squares_gradient(v)    # v의 그래디언트 계산
    v = gradient_step(v, grad, -0.01)    # 그래디언트의 음수만큼 이동
    print(epoch, v)
    
assert distance(v, [0, 0, 0]) < 0.001    # v는 0에 수렴해야 함


# Chapter 08 _ 8.5 경사 하강법으로 모델 학습
# 경사하강법으로 손실을 최소화하는 모델의 파라미터를 구하기
# x는 -50 ~ 49 사이의 값이며, y는 항상 20 * x + 5
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

# 한개의 데이터 포인트에서 오차의 그래디언트를 계산해주는 함수
def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept    # 모델의 예측값
    error = (predicted - y)              # 오차는 (예측값 - 실제값)
    squared_error = error ** 2           # 오차의 제곱을 최소화하자
    grad = [2 * error * x, 2 * error]    # 그래디언트 사용
    return grad

# 평균제곱오차의 그래디언트
# 1. 임의의 theta로 시작
# 2. 모든 그래디언트의 평균 계산
# 3. theta를 2번에서 계산된 값으로 변경
# 4. 반복

from scratch.linear_algebra import vector_mean
    
# 임의의 경사와 절편으로 시작
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
learning_rate = 0.001
    
for epoch in range(5000):
    # 모든 그래디언트의 평균 계산
    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    # 그래디언트만큼 이동
    theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)
    
slope, intercept = theta
assert 19.9 < slope < 20.1,   "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"


# Chapter 08 _ 8.6 미니배치와 SGD(Stochastic Gradient Descent)
# 미니배치 경사하강법(minibatch gradient descent)에서는
# 전체 데이터 셋의 샘플인 미니배치에서 그래디언트를 계산함
from typing import TypeVar, List, Iterator

T = TypeVar('T')                # 변수의 타입과 무관한 함수를 생성

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """dataset에서 batch_size만큼 데이터 포인트를 샘플링해서 미니배치를 생성"""
    # 각 미니배치의 시작점인 0, batch_size, 2 * batch_size, ...을 나열
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts)  # 미니배치의 순서를 섞는다.

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]

# 앞서 살펴본 예시를 미니배치로 다시 풀어보자
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
for epoch in range(1000):
    for batch in minibatches(inputs, batch_size=20):
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)
    
slope, intercept = theta
assert 19.9 < slope < 20.1,   "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"

# SGD의 경우에는 각 에포크마다 단 하나의 데이터 포인트에서 그래디언트를 계산함
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
for epoch in range(100):
    for x, y in inputs:
        grad = linear_gradient(x, y, theta)
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)
    
slope, intercept = theta
assert 19.9 < slope < 20.1,   "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"
