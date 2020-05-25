# Chapter 07 _ 7.1 통계적 가설검정
# 가설이란, '이 동전은 앞뒤가 나올 확률이 공평한 동전이다.'
#          '데이터 과학자는 R보다 파이썬을 선호한다.'
#          '닫기 버튼이 작아서 찾기 힘든 광고 창의 띄우면 사용자는 해당 사이트를
#           죽었다 깨어나도 들어가지 않을 것이다.'
# 등과 같은 주장을 의미
# 기존의 입장을 나타내는 귀무가설(== 영가설, H0)
# 이와 대비되는 입장을 나타내는 대립가설(H1)
# 을 통계적으로 비교해서 귀무가설을 기각할지 말지 결정함


# Chapter 07 _ 7.2 예시: 동전 던지기

from typing import Tuple
import math

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Binomial(n, p)에 해당되는 mu(평균)와 sigma(표준편차) 계산"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

from scratch.probability import normal_cdf

# 누적 분포 함수는 확률변수가 특정 값보다 작을 확률을 나타냄
normal_probability_below = normal_cdf

# 만약 확률 변수가 특정 값보다 작지 않다면, 특정 값보다 크다는 것을 의미함
def normal_probability_above(lo: float,
                             mu: float = 0,
                             sigma: float = 1) -> float:
    """mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo보다 클 확률"""
    return 1 - normal_cdf(lo, mu, sigma)

# 만약 확률변수가 hi보다 작고 lo보다 작지 않다면 확률변수는 hi와 lo 사이에 존재함
def normal_probability_between(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo와 hi 사이에 있을 확률"""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# 만약 확률변수가 범위 밖에 존재한다면 범위 안에 존재하지 않는다는 것을 의미
def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo와 hi 사이에 없을 확률"""
    return 1 - normal_probability_between(lo, hi, mu, sigma)

# 확률이 주어졌을 때 평균을 중심으로 하는 (대칭적) 구간을 구할 수도 있음
# 예로, 분포의 60%를 차지하는 평균 중심의 구간을 구하고 싶다면
# 양쪽 꼬리 부분이 각각 분포의 20%를 차지하는 지점을 구하면 됨
from scratch.probability import inverse_normal_cdf

def normal_upper_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """P(Z <= z) = probability인 z값을 반환"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """P(Z <= z) = probability인 z값을 반환"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float,
                            mu: float = 0,
                            sigma: float = 1) -> Tuple[float, float]:
    """
    입력한 probability 값을 포함하고
    평균을 중심으로 대칭적 구간을 반환
    """
    tail_probability = (1 - probability) / 2

    # 구간의 상한은 tail_probability 값 이상의 확률 값을 갖고 있다.
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # 구간의 하한은 tail_probability 값 이하의 확률 값을 갖고 있다.
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

# (469, 531)
lower_boun, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# p가 정말로 0.5, 즉 H0(귀무가설)이 참이라면, X가 주어진 범위를 벗어날 확률은
# 우리가 원한 대로 5% 밖에 되지 않을 것임
# 이것은, 만약 귀무가설이 참이라면 이 가설검정은 20번 중 19번은 올바른 결과를 줄 것이라는 의미

# 제 2종 오류를 범하지 않을 확률을 구하면 검정력(power)를 얻을 수 있음
# 제 2종 오류란, 귀무가설이 거짓이지만 귀무가설을 기각하지 않는 오류
# 제 2종 오류를 측정하기 위해서는 먼저 귀무가설이 거짓이라는 것이 무엇을 의미하는지 알아야 한다
# (p가 0.5가 아니라는 말은 X의 분포에 관해 많은 것을 알려주지는 않는다)
# 예로, p = 0.55, 즉 동전의 앞면이 나올 확률이 약간 편향되어 있다면 검정력은 다음과 같다

# p = 0.5라고 가정할 때, 유의수준이 5%인 구간
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# p = 0.55인 경우의 실제 평균과 표준편차
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# 제 2종 오류란 귀무가설을 기각하지 못한다는 의미
# 즉, X가 주어진 구간 안에 존재할 경우를 의미
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability

# 한편 p <= 0.5, 즉 동전이 앞면에 편향되지 않는 경우를 귀무가설로 정한다면
# X가 500보다 크면 귀무가설을 기각하고, 500보다 작다면 기각하지 않는 단측검정(one-sided test)이 필요
# 유의수준이 5%인 가설검정을 위해서는 normal_probability_below를 사용하여
# 분포의 95%가 해당 값 이하인 경계 값을 찾을 수 있음

hi = normal_upper_bound(0.95, mu_0, sigma_0)
# 결과값은 526 (< 531, 분포 상위 부분에 더 높은 확률을 주기 위해)

type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability


# Chapter 07 _ 7.3 p-value
# 가설을 바라보는 또 하나의 관점, p-value
# 이것은 어떤 확률값을 기준으로 구간을 선택하는 대신에 귀무가설이 참이라고 가정하고
# 실제로 관측된 값보다 더 극단적인 값이 나올 확률을 구하는 것
# 동전이 공평한지를 확인해보기 위해 양측검정을 해보자

def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    mu(평균)와 sigma(표준편차)를 따르는 정규분포에서
    x같이 극단적인 값이 나올 확률은 얼마나?
    """
    if x >= mu:
        # 만약 x가 평균보다 크다면 x보다 큰 부분이 꼬리
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # 만약 x가 평균보다 작다면 x보다 작은 부분이 꼬리
        return 2 * normal_probability_below(x, mu, sigma)

two_sided_p_value(529.5, mu_0, sigma_0)

# 시뮬레이션
import random

extreme_value_count = 0
for _ in range(1000):
    num_heads = sum(1 if random.random() < 0.5 else 0    # 앞면이 나온 경우
                    for _ in range(1000))                # 동전을 1000번 던져서
    if num_heads >= 530 or num_heads <= 470:             # 그리고 극한 값이
        extreme_value_count += 1                         # 몇 번 나오는지 세어봄

# p-value was 0.062 => ~62 extreme values out of 1000
assert 59 < extreme_value_count < 65, f"{extreme_value_count}"

# 계산된 p-value가 5%보다 크기 때문에 귀무가설을 기각핮 ㅣ않는다
# 만약 동전의 앞면이 532번 나왔다면 p-value는 5%보다 작을 것이고,
# 이 경우 귀무가설을 기각한다

two_sided_p_value(531.5, mu_0, sigma_0)

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

# 동전의 앞면이 525번 나왔다면 단측검정을 위한 p-value는 다음과 같이 계산되며
# 귀무가설을 기각하지 않을 것이다
upper_p_value(524.5, mu_0, sigma_0)     # 0.061

# 만약 동전의 앞면이 527번 나왔다면 p-value는 다음과 같이 계산되며
# 귀무가설을 기각할 것이다
upper_p_value(526.5, mu_0, sigma_0)     # 0.047


# Chapter 07 _ 7.4 신뢰구간
# 사건에 대한 분포를 모를 때, 관측된 값에 대한 '신뢰구간'을 사용하여
# 가설을 검정함
math.sqrt(p * (1 - p) / 1000)

# 동전을 1,000번 던져서 앞면이 525번 나온 경우
p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)       # 0.0158
normal_two_sided_bounds(0.95, mu, sigma)            # [0.4940, 0.5560]

# 동전을 1,000번 던져서 앞면이 540번 나온 경우
p_hat = 540 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)       # 0.0158
normal_two_sided_bounds(0.95, mu, sigma)            # [0.5091, 0.5709]


# Chapter 07 _ 7.5 p 해킹
# 귀무가설을 잘못 기각하는 경우가 5%인 가설 검정은
# 정의에서 알 수 있듯, 모든 경우의 5%에서 귀무가설을 잘못 기각함
from typing import List

def run_experiment() -> List[bool]:
    # 동전을 1000번 던져서 True = 앞, False = 뒷면
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment: List[bool]) -> bool:
    # 신뢰구간을 5%로 설정
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

random.seed(0)
experiments = [run_experiment() for _ in range(1000)]
num_rejections = len([experiment
                      for experiment in experiments
                      if reject_fairness(experiment)])

assert num_rejections == 46

# 즉 "의미 있는" 결과를 찾으려면 보통 의미 있는 결과를 찾을 수 있다는 것
# 주어진 데이터에 대해 다양한 가설들을 검정하다 보면
# 이중 하나는 반드시 의미 있는 가설로 보일 수 있다
# 여기서 적절히 이상치를 제거하면 0.05 이하의 p-value를 구할 수 있을 것이다
# 이렇게 p-value의 관점에서 추론을 하면 'p 해킹'이 발생할 수 있다
# 예로, "지구는 둥글다"라는 기사는 이러한 p 해킹의 문제점을 잘 설명해줌
"""
데이터 과학을 잘하기 위한 세 가지
1) 가설은 데이터를 보기 전에 세운다
2) 데이터를 전처리 할 때는 세워둔 가설을 잠시 잊는다
3) p-value가 전부는 아니다(대안으로 베이즈 추론을 사용할 수 있음)
"""


# Chapter 07 _ 7.6 예시: A/B test 해보기
# 예로, 광고 A와 광고 B가 있을 때
# 광고 A를 본 1000명 중 990명이 광고를 클릭했고
# 광고 B를 본 1000명 중 10명이 광고를 클릭했따면 명백히 광고 A가 더 좋지만
# 이러한 명확한 차이가 없다면 통계적 추론을 통해 인사이트를 얻어야 한다

# NA명의 사용자가 광고 A를 보았고, 그중 nA명이 광고를 클릭함
# 각 사용자가 광고를 보고 클릭하는 것을 베르누이 시행으로 볼 수 있고
# 각 사용자가 광고 A를 클릭할 확률을 pA라고 정의
# 그렇다면, nA/NA는 평균이 pA이고 표준편차가 sigmaA = sqrt(p * (1 - p) / NA)
# 위와 같은 정규분포에 근접함
# 광고 B의 경우도 같다
# 만약 두 정규분포가 독립이라면, 두 정규분포의 차이 또한 평균이 pB - pA이고
# 표준편차가 sqrt(sigmaA^2 + sigmaB^2)인 정규분포를 따르게 된다
def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

# pA와 pB가 같다는(즉, pA - pB = 0) 귀무가설은 다음의 통계치로 검정 가능
def a_b_test_statistics(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

# 위 식은 대략 표준정규분포를 따름

z = a_b_test_statistics(1000, 200, 1000, 180)   # -1.14
two_sided_p_value(z)                            # 0.254

z = a_b_test_statistics(1000, 200, 1000, 150)   # -2.94
two_sided_p_value(z)                            # 0.003


# Chapter 07 _ 7.7 베이즈 추론
# 알려지지 않은 파라미터를 확률변수로 보는 방법
# 사전분포가 주어지고, 관측된 데이터와 베이즈 정리를 사용하여 사후분포를 갱신할 수 있다
def B(alphaL: float, beta: float) -> float:
    """모든 확률값의 합이 1이 되도록 해주는 정규화 값"""
    return math.gamma(alpah) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1:        # [0, 1] 구간 밖에서는 밀도가 없다
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)

# 일반적으로 베타분포의 중심은 다음과 같다
alpha / (alpha + beta)
