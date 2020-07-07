import tensorflow as tf

## Tensorflow 상의 더하기
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)            # 그냥 더하기(+)하면 에러난다.
print(f'node1 : {node1}, node2 : {node2} \nnode3 : {node3}')
# node1 : Tensor("Const:0", shape=(), dtype=float32), node2 : Tensor("Const_1:0", shape=(), dtype=float32)
# node3 : Tensor("Add:0", shape=(), dtype=float32)

## Session()
sess = tf.Session()
print(f'sess.run(node1, node2) : {sess.run(node1)}, {sess.run(node2)} \nsess.run(node3) : {sess.run(node3)}')