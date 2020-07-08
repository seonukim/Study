# mv : Multi Variables

import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)

y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'weight1')
w2 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'weight2')
w3 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'weight3')
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

cost = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y_data))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 45e-6)
train = optimizer.minimize(cost)


fedict = {x1: x1_data, x2: x2_data, x3: x3_data, y: y_data}
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for step in range(2000 + 1):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = fedict)

        if step % 50 == 0:
            print(f'{step}, cost : {cost_val}, \n{step} H : {hy_val}\n')