import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from comprehension.parameter import user_params

'''
  * Created by linsu on 2018/8/16.
  * mailto: lsishere2002@hotmail.com
'''

class BiLstmLayer(tf.layers.Layer):
    def __init__(self, params:user_params,feature_size:int, is_trainning=True, name="BiLstmLayer", dtype=tf.float32):
        super(BiLstmLayer, self).__init__(is_trainning, name, dtype)
        self.params = params
        self.feature_size = feature_size
        self.output_feature_size = self.params.rnn_hidden_size*2
        self.keep_pro = 1.0 if not self.trainable else 1.0-self.params.drop_out_rate

    def get_output_feature_size(self):
        return self.output_feature_size

    def build(self, _):
        self.fw_cell = tf.nn.rnn_cell.LSTMCell(
            self.params.rnn_hidden_size
            ,initializer=xavier_initializer()
            ,name = "fw_cell"
        )
        self.bw_cell = tf.nn.rnn_cell.LSTMCell(
            self.params.rnn_hidden_size
            ,initializer=xavier_initializer()
            ,name = "bw_cell"
        )

        self.fw_cell = tf.nn.rnn_cell.DropoutWrapper(self.fw_cell,input_keep_prob=self.keep_pro)
        self.bw_cell = tf.nn.rnn_cell.DropoutWrapper(self.bw_cell,input_keep_prob=self.keep_pro)
        # self.losses.extend(self.fw_cell.losses)
        # self.losses.extend(self.bw_cell.losses)
        self.built = True

    def call(self, inputs, **kwargs):
        #[batch_size*sent_number,seq_len,embedding_size]
        context = inputs[self.params.context_name]
        #[batch_size,seq_len,embedding_size]
        question = inputs[self.params.question_name]

        #[batch_size*sent_number]
        context_len = inputs["context_length"]
        #[batch_size]
        question_len = inputs["question_length"]
        #feature_size =self.feature_size
        output_context,_ = tf.nn.bidirectional_dynamic_rnn(self.fw_cell
                                                   ,self.bw_cell
                                                   ,context
                                                   ,sequence_length=context_len
                                                   ,dtype=tf.float32)
        #tf.get_variable_scope().reuse_variables()
        output_question,_ = tf.nn.bidirectional_dynamic_rnn(self.fw_cell
                                                           ,self.bw_cell
                                                           ,question
                                                           ,sequence_length=question_len
                                                           ,dtype=tf.float32)
        #[batch_size*sent_number,seq_len,2*hidden_size]
        context = tf.concat(output_context,-1)
        #[batch_size,q_seq_len,2*hidden_size]
        question = tf.concat(output_question,-1)
        inputs[self.params.context_name] = context
        inputs[self.params.question_name] = question

        return inputs