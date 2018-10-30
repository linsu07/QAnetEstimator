import math

import tensorflow as tf

from comprehension.parameter import user_params


class QAOutputLayer(tf.layers.Layer):
    def __init__(self, params:user_params,feature_size:int, is_trainning=True, name:str="QAOutputLayer", dtype=tf.float32):
        super(QAOutputLayer, self).__init__(is_trainning, name, dtype)
        self.params = params
        self.feature_size = feature_size
        self.exp_epsilon = -1e25
        self.scale = 1/math.sqrt(2*self.feature_size)

    def get_output_feature_size(self):
        return None

    def build(self, _):
        self.w_p1 = self.add_variable("w_p1",[2*self.feature_size],regularizer=tf.nn.l2_loss)
        self.w_p2 = self.add_variable("w_p2",[2*self.feature_size],regularizer=tf.nn.l2_loss)
        self.built = True

    def call(self, inputs, **kwargs):
        m0,m1,m2 = inputs["M0"],inputs["M1"],inputs["M2"]
        #[batch_size*sent_number,c_seq_len]
        context_mask = tf.squeeze(inputs["context_mask"],-1)

        start_context = tf.concat([ m0,m1],axis=-1)

        #[batch_size,sent_number,c_seq_len]
        start_factor = tf.reduce_sum(start_context*self.w_p1,axis=-1)
        # start_factor = tf.where(tf.cast(start_factor,bool),start_factor,tf.ones_like(start_factor)*self.exp_epsilon)
        start_factor = start_factor* self.scale
        padding = (1-context_mask)*self.exp_epsilon
        start_factor = start_factor+padding
        start_factor = tf.reshape(start_factor,[self.params.cur_batch_size,self.params.sent_number,self.params.c_seq_len])

        end_context = tf.concat([ m0,m2],axis=-1)
        #[batch_size,sent_number,c_seq_len]
        end_factor = tf.reduce_sum(end_context*self.w_p2,axis=-1)
        end_factor = end_factor*self.scale
        #end_factor = tf.where(tf.cast(end_factor,bool),end_factor,tf.ones_like(end_factor)*self.exp_epsilon)
        end_factor = end_factor+padding
        end_factor = tf.reshape(end_factor,[self.params.cur_batch_size,self.params.sent_number,self.params.c_seq_len])

        return start_factor,end_factor
