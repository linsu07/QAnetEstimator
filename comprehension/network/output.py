import math

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from comprehension.network.modeling import ModelingLayer
from comprehension.parameter import user_params
'''
  * Created by linsu on 2018/8/25.
  * mailto: lsishere2002@hotmail.com
'''
class OutputLayer(tf.layers.Layer):
    def __init__(self, params:user_params,feature_size:int, is_trainning=True, name:str="OutputLayer", dtype=tf.float32):
        super(OutputLayer, self).__init__(is_trainning, name, dtype)
        self.params = params
        self.feature_size = feature_size
        self.exp_epsilon = -1e25
        self.scale = 1/math.sqrt(self.feature_size)

    def get_output_feature_size(self):
        return None

    def build(self, _):
        self.w_p1 = self.add_variable("w_p1",[self.feature_size])
        self.w_p2 = self.add_variable("w_p2",[self.feature_size])
        self.bilstm = ModelingLayer(self.params,2*self.params.rnn_hidden_size,2,self.trainable,name="m2_lstm")
        self.built = True

    def call(self, inputs, **kwargs):
        #[batch_size*sent_number,c_seq_len,10*self.params.rnn_hidden_size]
        start_context = tf.concat([inputs[self.params.context_name],inputs["M"]],axis=-1)

        #[batch_size*sent_number,c_seq_len]
        start_factor = tf.reduce_sum(start_context*self.w_p1,axis=-1)
        start_factor = start_factor*self.scale
        #[batch_size*sent_number,c_seq_len]
        context_mask = tf.squeeze(inputs["context_mask"],-1)
        #start_factor = tf.where(tf.cast(start_factor,bool),start_factor,tf.ones_like(start_factor)*self.exp_epsilon)
        #start_factor = tf.reshape(start_factor,[-1,self.params.sent_number*self.params.c_seq_len])
        #start_p= tf.nn.softmax(start_factor,-1),

        padding = (1-context_mask)*self.exp_epsilon
        start_factor = start_factor+padding

        start_factor = tf.reshape(start_factor,[self.params.cur_batch_size,self.params.sent_number,self.params.c_seq_len])
        #[batch_size*sent_number,c_seq_len,10*hidden_sized]
        end_context = self.get_end_context(inputs,start_factor)
        # end_context = get_end_context_attention2(inputs,start_factor)

        end_factor = tf.reduce_sum(end_context*self.w_p2,axis=-1)
        end_factor = end_factor * self.scale
        #end_factor= tf.where(tf.cast(end_factor,bool),end_factor,tf.ones_like(end_factor)*-1e25)
        end_factor = end_factor+padding
        end_factor = tf.reshape(end_factor,[self.params.cur_batch_size,self.params.sent_number,self.params.c_seq_len])
        #end_factor = tf.reshape(end_factor,[-1,self.params.sent_number*self.params.c_seq_len])
        #end_p= tf.nn.softmax(end_factor,-1)

        return start_factor,end_factor

    def get_end_context(self,inputs,start_factor):
        context_len = inputs["context_length"]
        tmp_input = {self.params.context_name:inputs["M"],"context_length":context_len}
        m2 = self.bilstm(tmp_input)["M"]
        #[batch_size*sent_number,c_seq_len,10*hidden_sized]
        end_context = tf.concat([inputs[self.params.context_name],m2],axis=-1)
        return end_context

        
