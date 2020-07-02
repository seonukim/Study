# activation - LeakyReLU

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-6, 6, 0.01)

def leakyrelu_func(x):      # Leaky ReLU(Rectified Linear Unit)
    return (x >= 0)*x + (x < 0)*0.01*x  # 알파값(보통 0.01) 조정 가능
    # return np.maximum(0.01*x, x)  same

plt.plot(x, leakyrelu_func(x), linestyle = '--', label = 'Leaky ReLU')
plt.show()