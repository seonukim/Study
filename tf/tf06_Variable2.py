# 변수 출력의 여러가지 방법
# 1) tf.compat.v1.Session()
# 2) tf.compat.v1.InteractiveSession()  -> .eval()
# 3) tf.compat.v1.Session()     -> .eval(session = sess)

import warnings ; warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.set_random_seed(777)
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()

W = tf.compat.v1.Variable(tf.random.normal([1]), name = 'weight')     # 랜덤 정규분포
b = tf.compat.v1.Variable(tf.random.normal([1], name = 'bias'))       # [] 안의 숫자는 shape를 의미한다
print(W)        # <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>


### 변수 출력의 여러 방법들
W = tf.compat.v1.Variable([0.3], tf.float32)        # 변수 수정
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
aaa = sess.run(W)
print(f'aaa: {aaa}')
sess.close()            # Session을 열면 항상 close 해줘야한다, with문을 쓰면 자동으로 close()를 호출함

## tf.compat.v1.InteractiveSession()            일반 Session()과 결과는 같지만, 사용법이 다르다
sess = tf.compat.v1.InteractiveSession()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
bbb = W.eval()          # InteractiveSession()을 사용하면 sess.run()을 하지 않고 .eval()로 함
print(f'bbb: {bbb}')
sess.close()

## tf.compat.v1.Session()의 .eval() 사용법
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
ccc = W.eval(session = sess)            # Session()도 .eval()을 사용할 수 있지만, session = sess로 파라미터를 전달해야한다
print(f'ccc: {ccc}')
sess.close()