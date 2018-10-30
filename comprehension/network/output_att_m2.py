import tensorflow as tf
from comprehension.network.modeling import ModelingLayer
from comprehension.network.output import OutputLayer
from comprehension.parameter import user_params
'''
  * Created by linsu on 2018/8/25.
  * mailto: lsishere2002@hotmail.com
'''
class OutputLayerAttM2(OutputLayer):
    def __init__(self, params:user_params,feature_size:int, is_trainning=True, name:str="OutputLayerAttM2", dtype=tf.float32):
        super(OutputLayerAttM2, self).__init__(params,feature_size,is_trainning, name, dtype)


    def get_end_context(self,inputs,start_factor):
        #[batch_size*sent_number]
        context_len = inputs["context_length"]
        #[batch_size*sent_number,c_seq_len,8*hidden_size]
        G = inputs[self.params.context_name]
        #[batch_size,1,sent_number*seq_len]
        start_factor = tf.nn.softmax(tf.reshape(start_factor,[-1,1,self.params.sent_number*self.params.c_seq_len]),-1)
        #[batch_size,sent_number*seq_len,2*hidden_size]
        start_context = tf.reshape(inputs["M"],[-1,self.params.sent_number*self.params.c_seq_len,2*self.params.rnn_hidden_size])

        #[batch_size,1,2*hidden_size]
        matmul = tf.matmul(start_factor,start_context)
        matmul = tf.reshape(matmul,[-1,1,1,2*self.params.rnn_hidden_size])
        #[-1,self.params.sent_number,self.params.c_seq_len,2*self.params.rnn_hidden_size]
        matmul = tf.tile(matmul,[1,self.params.sent_number,self.params.c_seq_len,1])
        matmul = tf.reshape(matmul,[-1,self.params.c_seq_len,2*self.params.rnn_hidden_size])


        #[batch_size*sent_number,c_seq_len,2*self.params.rnn_hidden_size]
        matmul = inputs["context_mask"]* matmul

        context = tf.concat([G,inputs["M"],matmul,inputs["M"]*matmul],axis=-1)

        tmp_input = {self.params.context_name:context,"context_length":context_len}
        self.bilstm.feature_size = 14*self.params.rnn_hidden_size
        m2 = self.bilstm(tmp_input)["M"]
        end_context = tf.concat([G,m2],axis=-1)
        return end_context
