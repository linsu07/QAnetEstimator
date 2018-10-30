import tensorflow as tf

tf.enable_eager_execution()

x = tf.constant([1,2,3,4])
x = tf.cast(x,dtype=tf.float32)

#-1.5,-0.5,0.5,1.5

2.25,0.25+0.25,2.25
mean,v = tf.nn.moments(x,-1,keep_dims=True)

print(mean)

print(v)