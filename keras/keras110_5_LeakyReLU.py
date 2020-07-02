# activation - LeakyReLU

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-6, 6, 0.01)

def leakyrelu(x):      # Leaky ReLU(Rectified Linear Unit)
    return np.maximum(0.1 * x, x)  #same

plt.plot(x, leakyrelu(x), linestyle = '--', label = 'Leaky ReLU')
plt.show()