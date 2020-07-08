# mv : Multi Variables

import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x_data = [[73., 51., 65.],
          [92., 98., 11.],
          [89., 31., 33.],
          [99., 33., 100.],
          [17., 66., 79.]]
y_data = [[152.],
          [185.],
          [180.],
          [205.],
          [142.]]

x = tf.compat.v1.placeholder(tf.float32, shape = [None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])
w = tf.compat.v1.Variable(tf.random.normal([3, 1]), name = 'weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

hypothesis = tf.compat.v1.add(tf.compat.v1.matmul(x, w), b)      # wx + b

cost = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y_data))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 5e-6)
train = optimizer.minimize(cost)


fedict = {x: x_data, y: y_data}
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for step in range(2000 + 1):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = fedict)

        if step % 50 == 0:
            print(f'{step}, cost  : {cost_val}, \n{step} 예측값 : \n{hy_val}\n')