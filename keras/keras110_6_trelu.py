# activation - TReLU

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-6, 6, 0.01)

def trelu_func(x):
    return (x > 1) * x    # 임계값 (1) 조정 가능

plt.plot(x, trelu_func(x), linestyle = '--', label = "Threshholded ReLU")
plt.show()