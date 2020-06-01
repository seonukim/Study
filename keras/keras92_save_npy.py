import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

print(iris)

x_data = iris.data
y_data = iris.target

print(type(x_data))
print(type(y_data))

print(x_data.shape)
print(y_data.shape)

# save file by using numpy (.npy file)
np.save('./data/iris_x.npy', arr=x_data)
np.save('./data/iris_y.npy', arr=y_data)

# load npy file
x_data_load = np.load('./data/iris_x.npy')
y_data_load = np.load('./data/iris_y.npy')

# print(x_data_load)
# print(y_data_load)
# print(x_data_load.shape)
# print(y_data_load.shape)