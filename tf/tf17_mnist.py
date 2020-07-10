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

x_train = x_train.reshape(-1, 28*28).astype('float32')/255.
x_test = x_test.reshape(-1, 28*28).astype('float32')/255.

print(x_train.shape)   # (60000, 784)
print(y_train.shape)   # (60000, 10) 

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size)  # 60000 / 100

x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])

keep_prob = tf.placeholder(tf.float32)        # dropout


                                # input / output   
# w = tf.variable(tf.random_normal([784, 512]), name = 'weight1')  # 동일  
w1 = tf.get_variable('w1', shape=[784, 512],                        # 초기 변수가 없으면 알아서 할당함/ 파라미터 많음
                    initializer=tf.contrib.layers.xavier_initializer())                        
b1 = tf.Variable(tf.random_normal([512]), name = 'bias')
L1 = tf.nn.selu(tf.matmul(x, w1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)   

w2 = tf.get_variable('w2', shape=[512, 512],                       
                    initializer=tf.contrib.layers.xavier_initializer())                        
b2 = tf.Variable(tf.random_normal([512]), name = 'bias')
L2 = tf.nn.selu(tf.matmul(L1, w2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

w3 = tf.get_variable('w3', shape=[512, 512],                       
                    initializer=tf.contrib.layers.xavier_initializer())                        
b3 = tf.Variable(tf.random_normal([512]), name = 'bias')
L3 = tf.nn.selu(tf.matmul(L2, w3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

w4 = tf.get_variable('w4', shape=[512, 256],                       
                    initializer=tf.contrib.layers.xavier_initializer())                        
b4 = tf.Variable(tf.random_normal([256]), name = 'bias')
L4 = tf.nn.selu(tf.matmul(L3, w4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

w5 = tf.get_variable('w5', shape=[256, 10],                       
                    initializer=tf.contrib.layers.xavier_initializer())                        
b5 = tf.Variable(tf.random_normal([10]), name = 'bias')
hypothesis = tf.nn.softmax(tf.matmul(L4, w5) + b5)                           # 마지막 output / dropout필요 없음


loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):            # 15
    ave_cost = 0

    for i in range(total_batch):                # 600
        start = i*batch_size
        end = start + batch_size

        batch_xs, batch_ys = x_train[start : end], y_train[start : end]
        
        feed_dict = {x:batch_xs, y:batch_ys, keep_prob:0.9}           # (1 - keep_prob)만큼 dropout한다.
        c, _=sess.run([loss, optimizer], feed_dict=feed_dict) 
        ave_cost += c / total_batch

    print('Epoch :', '%04d'%(epoch+1),
          'loss =', '{:.9f}'.format(ave_cost))

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc :', sess.run(accuracy, feed_dict = {x:x_test, y:y_test, keep_prob:1}))
