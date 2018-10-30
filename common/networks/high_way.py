import tensorflow as tf

'''
  * Created by linsu on 2018/8/16.
  * mailto: lsishere2002@hotmail.com
'''


class HighwayLayer(tf.layers.Layer):
    def __init__(self,feature_size:int,layers_number=1,dropout_rate = 0.0, is_trainning=False, name="highway_layer", dtype=tf.float32):
        super(HighwayLayer, self).__init__(is_trainning, name, dtype)
        self.feature_size = feature_size
        self.layers_number = layers_number
        self.dropout_rate = dropout_rate
    def build(self, _):
        self.gates = [tf.layers.Dense(self.feature_size,activation=tf.sigmoid,bias_initializer=tf.constant_initializer(-2)
                                 ,name = "gate_{}".format(i),kernel_regularizer=tf.nn.l2_loss)
                 for i in range(self.layers_number)]
        self.transforms = [tf.layers.Dense(self.feature_size,activation=tf.nn.relu,bias_initializer=tf.zeros_initializer()
                                     ,name = "trans_{}".format(i),kernel_regularizer=tf.nn.l2_loss)
                     for i in range(self.layers_number)]
        self.built = True

    def call(self, input, **kwargs):
        mask = kwargs.get("mask")
        output = input
        for i in range(self.layers_number):
            output = self.highway(output,i)
        if mask!=None:
            output = output*mask
        return output

    def highway(self,input,i):
        t_gate = self.gates[i](input)
        c_gate = 1.0- t_gate
        transform = self.transforms[i](input)
        transform = tf.layers.dropout(transform,self.dropout_rate,training=self.trainable)
        return tf.add(tf.multiply(t_gate,transform) ,tf.multiply(c_gate,input))