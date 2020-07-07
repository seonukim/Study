import tensorflow as tf
print(tf.__version__)

hello = tf.constant("Hello World!")     # constant : 상수; 바뀌지 않는 수
print(hello)                            # Tensor("Const:0", shape=(), dtype=string) <- Tensor의 자료형

'''
어떤 값을 우리가 인식할 수 있도록 보여지게 하려면
Session() 클래스를 사용해야 한다.
Session()은 텐서플로 알고리즘 내 연산을 하는 부분

Tensorflow 1. 대 버전에서는 항상 Session()을 사용한다.
'''
sess = tf.Session()
print(sess.run(hello))                  # b'Hello World!'