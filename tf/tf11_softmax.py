import tensorflow as tf
import numpy as np

seed = 77

tf.set_random_seed(seed)

x_data = np.array([[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]], dtype=np.float32)

y_data = np.array([[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]], dtype=np.float32)

print(x_data.shape)
print(y_data.shape)

x_col_num = x_data.shape[1] # 4
y_col_num = y_data.shape[1] # 3

x = tf.placeholder(tf.float32, shape=[None, x_col_num])
y = tf.placeholder(tf.float32, shape=[None, y_col_num])

w = tf.Variable(tf.random_normal([x_col_num, y_col_num]), name = 'weight')
b = tf.Variable(tf.random_normal([1, y_col_num]), name = 'bias') # y_col_num

h = tf.nn.softmax(tf.matmul(x,w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h),axis=1)) # loss ... 계산 방법 ...

opt = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss) # 어떻게 쓰는지 어떻게 계산하는지 지금은 일단 쓰고 시간이 많을때 꼭!!! 공부하라 경사하강법

# train = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        _, cost_val = sess.run([opt, loss], feed_dict={ x: x_data, y: y_data})

        if i % 200 == 0 :
            print( i , cost_val)
    
    pred = sess.run(h, feed_dict={x:[[1,11,7,9]]}) # keras model.predict(x_test_data)
    print(pred, sess.run(tf.argmax(pred, 1))) # tf.argmax(a, 1) 안에 값들중에 가장 큰 값의 인덱스를 표시하라

    all = sess.run(h, feed_dict={x: [[1,11,7,9], [1,15,5,9],[1,11,17,10]]})
    print(all, sess.run(tf.argmax(all, 1)))

    # feed_dict={x: [np.append(a, 0), np.append(b, 0), np.append(c, 0)]})