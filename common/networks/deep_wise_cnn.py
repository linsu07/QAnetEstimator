import  tensorflow as tf
'''
  * Created by linsu on 2018/9/12.
  * mailto: lsishere2002@hotmail.com
'''
class DeepWiseCnn(tf.layers.Layer):
    def __init__(self, kernel_size,feature_size,is_trainning=False,dropout_rate=0.0
               , name="DeepWiseCnn"
               , dtype=tf.float32):
        super(DeepWiseCnn,self).__init__(is_trainning, name, dtype)
        self.kernel_size = kernel_size
        self.feature_size = feature_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.deep_wise_filter = self.add_variable(name = "deep_wise_filter"
                                             ,shape = [self.kernel_size,1,self.feature_size,1]
                                             ,regularizer=tf.nn.l2_loss
                                             )
        self.point_wise_filter = self.add_variable(name = "point_wise_filter"
                                                   ,shape = [1,1,self.feature_size,self.feature_size]
                                                   ,regularizer=tf.nn.l2_loss)

        self.bias = self.add_variable(name="bias",shape = [self.feature_size],initializer=tf.zeros_initializer())
    """
    input 输入的shape必定为 batch_size*sent_number,seq_len,1,feature_size
    or
    batch_size,seq_len,1,feature_size
    """
    def call(self, input, **kwargs):
        mask = kwargs.get('mask')
        output = tf.nn.separable_conv2d(input,self.deep_wise_filter,self.point_wise_filter,
                                        strides=[1,1,1,1],padding="SAME")

        output = tf.nn.relu(output+self.bias)
        if mask!=None:
            output= output*tf.expand_dims(mask,-1)
        
        return output
