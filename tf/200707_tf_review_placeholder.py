import warnings
warnings.filterwarnings(action = 'ignore')
import tensorflow as tf

## 그래프 생성
g = tf.Graph()
with g.as_default():
    x = tf.compat.v1.placeholder(dtype = tf.float32,
                                 shape = (None), name = 'x')
    w = tf.Variable(2.0, name = 'weight')
    b = tf.Variable(0.7, name = 'bias')

    z = w * x + b

    init = tf.compat.v1.global_variables_initializer()      # 초기화 변수 선언

## 세션을 만들고 그래프 g를 전달
with tf.compat.v1.Session(graph = g) as sess:
    # w와 b를 초기화한다
    sess.run(init)

    # z를 평가한다
    for t in [1.0, 0.6, -1.8]:
        print(f'x = {t}, --> z = {sess.run(z, feed_dict = {x:t}):.1f}')
'''
x = 1.0,  --> z = 2.7
x = 0.6,  --> z = 1.9
x = -1.8, --> z = -2.9
'''

print(z)        # Tensor("add:0", dtype=float32)

## tf.placeholder() 사용해보기
se = tf.Session()
node1 = tf.placeholder(dtype = tf.float32)
node2 = tf.placeholder(dtype = tf.float32)

c = 3
d = 4

adder_res = node1 + node2

feeddic = {node1: c, node2: d}

res = se.run(adder_res, feed_dict = feeddic)
print(res)