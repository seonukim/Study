import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist          
import numpy as np               
mnist.load_data()                                         

(x_train, y_train), (x_test, y_test) = mnist.load_data() 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

print(x_train.shape)   # (60000, 28, 28, 1)
print(y_train.shape)   # (60000, 10) 

## 파라미터
learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size)                # 60000 / 100

x = tf.placeholder(tf.float32, shape = [None, 784])
x_img = tf.compat.v1.reshape(x, [-1, 28, 28, 1])            # tf의 reshape
y = tf.placeholder(tf.float32, shape = [None, 10])
keep_prob = tf.placeholder(tf.float32)                      # dropout


                                # input / output   
# w = tf.variable(tf.random_normal([784, 512]), name = 'weight1')  # 동일  
w1 = tf.get_variable('w1', shape = [3, 3, 1, 32],                           # 3, 3 : kernel_size, input_shape = 28, 28, 1, 32 = output_node
                    initializer = tf.contrib.layers.xavier_initializer())                        
L1 = tf.nn.conv2d(x_img, w1, strides = [1, 1, 1, 1], padding = 'SAME')                          # 'SAME' : 대문자로 !!
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')         # maxpooling 추가
L1 = tf.nn.dropout(L1, keep_prob = keep_prob)

w2 = tf.get_variable('w2', shape = [3, 3, 32, 64], initializer = tf.contrib.layers.xavier_initializer())                        
L2 = tf.nn.conv2d(L1, w2, strides = [1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
L2 = tf.nn.dropout(L2, keep_prob = keep_prob)
print(L2)           # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)     <- Maxpool 2번 적용
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])          # Flatten()과 같다, 쫙 펼치기: 다 곱한 값

w3 = tf.get_variable('w3', shape = [7 * 7 * 64, 10], initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]), name = 'bias')
hypothesis = tf.nn.softmax(tf.matmul(L2_flat, w3) + b3)

loss = tf.compat.v1.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):            # 15
    avg_cost = 0

    for i in range(total_batch):                # 600
        start = i * batch_size
        end = start + batch_size

        batch_xs, batch_ys = x_train[start : end], y_train[start : end]
        
        feed_dict = {x_img: batch_xs, y: batch_ys, keep_prob: 0.9}           # (1 - keep_prob)만큼 dropout한다.
        c, _ = sess.run([loss, optimizer], feed_dict = feed_dict) 
        avg_cost += c / total_batch

    print('Epoch :', '%04d' %(epoch + 1),
          'loss =', '{:.9f}'.format(avg_cost))

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc :', sess.run(accuracy, feed_dict = {x_img: x_test, y: y_test, keep_prob: 1}))        # Acc : 0.9235
