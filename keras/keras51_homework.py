# 과제 2의 첫번째 답

# x = [1, 2, 3]
# x = x - 1
# print(x)

import numpy as np
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
'''
y = y - 1


from keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)
'''

# 과제 2의 두번째 답
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1, 1)
ohe.fit(y)
y = ohe.transform(y).toarray()

print(y)
print(y.shape)
