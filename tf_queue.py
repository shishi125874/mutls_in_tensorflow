import tensorflow as tf

q = tf.FIFOQueue(3, 'int32')
init = q.enqueue_many(([0, 1, 2],))

x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for a in range(5):
        v, a = sess.run([x, q_inc])
        print(v)
