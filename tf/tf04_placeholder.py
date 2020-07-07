import tensorflow as tf
sess = tf.Session()

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

a = tf.placeholder(tf.float32)          # input()과 비슷한 개념이다
b = tf.placeholder(tf.float32)

adder_node = a + b
print(sess.run(adder_node, feed_dict = {a:3, b:4.5}))           # feed_dict에 있는 값을 adder_node해서, 7.5
print(sess.run(adder_node, feed_dict = {a:[1, 3], b:[2, 4]}))   # feed_dict에 있는 값을 adder_node해서 [3. 7.]

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict = {a:3, b:4.5}))       # feed_dict에 있는 값을 adder_node * 3해서 22.5

# constant : 변하지 않는 값, 상수
# placeholder : input()과 비슷한 개념 ..
# placeholder는 처음에 값을 넣지 않는다, sess.run() 할 때 feed_dict를 통해 값을 넣어준다