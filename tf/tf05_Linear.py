import tensorflow as tf
tf.set_random_seed(777)

x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1], name = 'bias'))

hypothesis = x_train * W + b