import tensorflow as tf

## 데이터 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape)                    # (50000, 32, 32, 3)
print(x_test.shape)                     # (10000, 32, 32, 3)
print(y_train.shape)                    # (50000, 1)
print(y_test.shape)                     # (10000, 1)

## 데이터 전처리 - 피쳐: 정규화, 레이블: 원핫인코딩
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
print(x_train.shape)                    # (50000, 32, 32, 3)
print(x_test.shape)                     # (10000, 32, 32, 3)
print(y_train.shape)                    # (50000, 10)
print(y_test.shape)                     # (10000, 10)


## 파라미터(학습률, 에포크, 배치사이즈, 총배치)
learning_rate = 0.001
training_epochs = 20
batch_size = 100
total_batch = int(len(x_train) / batch_size)                # 50000 / 100

## 플레이스홀더 정의
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 3072])
x_img = tf.compat.v1.reshape(x, [-1, 32, 32, 3])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 10])
keep_prob = tf.compat.v1.placeholder(tf.float32)            # dropout


## 모델 구성
w1 = tf.compat.v1.get_variable('w1', shape = [3, 3, 3, 32],                           # 3, 3 : kernel_size, input_shape = 32, 32, 3, 32 = output_node
                    initializer = tf.contrib.layers.xavier_initializer())                        
L1 = tf.nn.conv2d(x_img, w1, strides = [1, 1, 1, 1], padding = 'SAME')
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')         # maxpooling 추가
L1 = tf.nn.dropout(L1, keep_prob = keep_prob)

w2 = tf.compat.v1.get_variable('w2', shape = [3, 3, 32, 16], initializer = tf.contrib.layers.xavier_initializer())
# 히든레이어의 channel은 상위 레이어의 아웃풋을 받는다.
L2 = tf.nn.conv2d(L1, w2, strides = [1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
L2 = tf.nn.dropout(L2, keep_prob = keep_prob)
print(L2)           # Tensor("dropout_1/mul_1:0", shape=(?, 8, 8, 16), dtype=float32)     <- Maxpool 2번 적용
L2_flat = tf.reshape(L2, [-1, 8 * 8 * 16])

w3 = tf.compat.v1.get_variable('w3', shape = [8 * 8 * 16, 10], initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.compat.v1.Variable(tf.random_normal([10]), name = 'bias')
hypothesis = tf.nn.softmax(tf.matmul(L2_flat, w3) + b3)

loss = tf.compat.v1.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)


## 모델 훈련
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):            # 15
    avg_cost = 0

    for i in range(total_batch):                # 500
        start = i * batch_size
        end = start + batch_size

        batch_xs, batch_ys = x_train[start : end], y_train[start : end]
        
        feed_dict = {x_img: batch_xs, y: batch_ys, keep_prob: 0.9}
        c, _ = sess.run([loss, optimizer], feed_dict = feed_dict) 
        avg_cost += c / total_batch

    print('Epoch :', '%04d' %(epoch + 1),
          'loss =', '{:.9f}'.format(avg_cost))

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc :', sess.run(accuracy, feed_dict = {x_img: x_test, y: y_test, keep_prob: 1}))        # Acc : 0.411
'''
Epoch : 0001 loss = 2.398788206
Epoch : 0002 loss = 2.251591799
Epoch : 0003 loss = 2.176436255
Epoch : 0004 loss = 2.107692084
Epoch : 0005 loss = 2.048826686
Epoch : 0006 loss = 2.005827047
Epoch : 0007 loss = 1.971908106
Epoch : 0008 loss = 1.942516550
Epoch : 0009 loss = 1.916806473
Epoch : 0010 loss = 1.887820534
Epoch : 0011 loss = 1.861951890
Epoch : 0012 loss = 1.835148171
Epoch : 0013 loss = 1.811636242
Epoch : 0014 loss = 1.786857782
Epoch : 0015 loss = 1.766578065
Epoch : 0016 loss = 1.751993315
Epoch : 0017 loss = 1.733022883
Epoch : 0018 loss = 1.720320500
Epoch : 0019 loss = 1.707591607
Epoch : 0020 loss = 1.695059067

Acc : 0.411
'''