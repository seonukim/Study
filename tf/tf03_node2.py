# 3 + 4 + 5 = x
# 4 - 3 = y
# 3 * 4 = z
# 4 / 2 = a

import tensorflow as tf
sess = tf.Session()

def sRun(input):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rst = input.eval()
        print(rst)
        return(rst)

print(sess.run(tf.constant(2)))
print(sess.run(tf.constant(3)))
print(sess.run(tf.constant(4)))
print(sess.run(tf.constant(5)))
# print(node2)
# print(node3)
# print(node4)
# print(node5)

print(sess.run(tf.add(tf.add(3, 4), 5)))          # 더하기    (tf.add_n([]) : 2개 이상의 값을 넣을 수 있다)
print(sess.run(tf.subtract(4, 3)))                # 빼기
print(sess.run(tf.multiply(3, 4)))                # 곱하기    (tf.matmul : 행렬의 곱셈)
print(sess.run(tf.divide(4, 2)))                  # 나누기    (tf.mod : 나머지)
