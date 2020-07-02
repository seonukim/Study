# 활성화 함수 - sigmoid
# activation의 목적 - 가중치 값을 한정시킨다.

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))         # np.exp() : 자연상수의 제곱
                                        # 밑이 자연상수 e인 지수함수 y = e^x로 변환해준다
x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

print(x.shape, y.shape)
plt.plot(x, y)
plt.grid()
# plt.show()

print(np.exp(5))
print(2.71828182845 ** 5)