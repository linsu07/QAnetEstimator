import  tensorflow as tf
import math

from comprehension.parameter import user_params
'''
  * Created by linsu on 2018/8/16.
  * mailto: lsishere2002@hotmail.com
'''

class BiDafLayer(tf.layers.Layer):
    def __init__(self, params:user_params,feature_size:int, is_trainning=True, name="BiDafLayer", dtype=tf.float32):
        super(BiDafLayer, self).__init__(is_trainning, name, dtype)
        self.params = params
        self.feature_size = feature_size
        self.exp_epsilon = -1e25
        self.scale = 1/math.sqrt(3*self.feature_size)
        self.output_feature_size = self.params.rnn_hidden_size*8

    def get_output_feature_size(self):
        return self.output_feature_size

    def build(self, _):
        self.alpha = self.add_variable("alpha",[3*self.feature_size])
        self.built = True

    def call(self, inputs, **kwargs):
        #[batch_size*sent_number,c_seq_len, feature_size]
        orig_context = inputs[self.params.context_name]
        #[batch_size,q_seq_len,feature_size]
        question = inputs[self.params.question_name]
        #[batch_size*sent_number,c_seq_len,q_seq_len feature_size]
        context = tf.tile(tf.expand_dims(orig_context,-2),[1,1,self.params.q_seq_len,1])
        #[batch_size,sent_number,c_seq_len,q_seq_len,feature_size]
        question = tf.tile(tf.expand_dims(tf.expand_dims(question,1),1),[1,self.params.sent_number,self.params.c_seq_len,1,1])
        #[batch_size*sent_number,c_seq_len,q_seq_len,feature_size]
        question = tf.reshape(question,[self.params.cur_batch_size*self.params.sent_number,self.params.c_seq_len,self.params.q_seq_len,-1])

        #[batch_size*sent_number,c_seq_len,q_seq_len,feature_size]
        c_q_mul = tf.multiply(context,question)
        #[batch_size*sent_number,c_seq_len,1]
        context_mask = inputs["context_mask"]
        #[batch_size,sent_number,c_seq_len,1]
        context_mask = tf.reshape(context_mask,[self.params.cur_batch_size,self.params.sent_number,self.params.c_seq_len,1])

        question_mask = tf.expand_dims(tf.squeeze(inputs["question_mask"],-1),-2)
        #[batch_size,1,1,q_seq_len]
        question_mask = tf.expand_dims(question_mask,-2)
        #[batch_size,sent_number,c_seq_len,q_seq_len]
        mask = tf.reshape(context_mask*question_mask,[self.params.cur_batch_size*self.params.sent_number,self.params.c_seq_len,self.params.q_seq_len])
        #[batch_size*sent_number,c_seq_len,q_seq_len]
        S = tf.reduce_sum(tf.concat([context,question,c_q_mul],axis=-1)*self.alpha,-1)
        S = S *self.scale
        #S = tf.where(tf.cast(S,tf.bool),S,tf.ones_like(S)*self.exp_epsilon)
        S = S+((1-mask)*self.exp_epsilon)
        #[batch_size*sent_number,c_seq_len,1,q_seq_len]
        factor_q = tf.expand_dims(tf.nn.softmax(S,axis = -1)* mask,-2)

        #[batch_size*sent_number,c_seq_len,feature_size],query remains at blank space
        new_query = tf.squeeze(tf.matmul(factor_q,question),axis=-2)

        #[batch_size*sent_number,c_seq_len]
        factor_c = tf.nn.softmax(tf.reduce_max(S,-1),axis=-1)
        #[batch_size*sent_number,1,c_seq_len]
        factor_c = tf.expand_dims(factor_c,-2)
        #[batch_size*sent_number,1,feature_size]
        q2c = tf.matmul(factor_c,orig_context)

        #[batch_size,,sent_number,c_seq_len,feature_size]
        #new_context = tf.tile(q2c,multiples=[1,1,self.params.c_seq_len,1])

        #[batch_size*sent_number,c_seq_len,4*feature_size]
        G = tf.concat([orig_context,new_query,tf.multiply(orig_context,new_query),tf.multiply(orig_context,q2c)],axis=-1)
        inputs[self.params.context_name] = G
        inputs[self.params.question_name] = None
        return inputs
