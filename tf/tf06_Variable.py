# hypothesis를 구하시오
# H = Wx + b
# aaa, bbb, ccc 자리에 각 hypothesis를 구하시오


import warnings ; warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x = [1., 2., 3.]
W = tf.compat.v1.Variable([0.3], tf.float32)
b = tf.compat.v1.Variable([1.0], tf.float32)

hypothesis = W * x + b

### Hypothesis 구해서 출력하기
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    hypothesis_1 = sess.run(hypothesis)

    print(f'Hypothesis_1: {hypothesis_1}')


with tf.compat.v1.InteractiveSession() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    hypothesis_2 = hypothesis.eval()

    print(f'Hypothesis_2: {hypothesis_2}')


'''
## tf.compat.v1.Session()의 .eval() 사용법
sess = tf.compat.v1.Session()
sess.run(init)
ccc = W.eval(session = sess)
print(f'ccc: {ccc}')
sess.close()
'''