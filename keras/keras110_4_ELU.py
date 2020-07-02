# activation - ELU

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def elu(x):
    x = np.copy(x)
    x[x < 0] = 0.2 * (np.exp(x[x < 0]) - 1)
    return x

plt.plot(x, elu(x), linestyle = '--', label = 'ELU')
plt.show()