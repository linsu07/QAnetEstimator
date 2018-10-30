import tensorflow as tf

if __name__=="__main__":

    x = [
        [1,3,5],
        [2,4,6],
        [1,1,1]
    ]
    y = [
        [1,3,5],
        [2,4,6],
        [2,1,1]
    ]

    z = [
        [1,3,5],
        [1,4,6],
        [2,1,1]
    ]
    x_t = tf.constant(x)
    y_t = tf.constant(y)
    z_t = tf.constant(z)
    acc,acc_op = tf.metrics.accuracy(x_t,y_t)
    acc1,acc_op1 = tf.metrics.accuracy(x_t,z_t,updates_collections=[acc_op])
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        init1 = tf.local_variables_initializer()
        sess.run([init,init1])
        a,ao = sess.run([acc,acc_op])
        print (a)
        print(ao)
        a1,ao1 = sess.run([acc1,acc_op1])
        print (a1)
        print(ao1)

