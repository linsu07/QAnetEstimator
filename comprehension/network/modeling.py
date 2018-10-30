import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from comprehension.parameter import user_params
'''
  * Created by linsu on 2018/8/19.
  * mailto: lsishere2002@hotmail.com
'''
class ModelingLayer(tf.layers.Layer):
    def __init__(self, params:user_params,feature_size:int, layer_number:int,is_trainning=True, name:str="modelingLayer", dtype=tf.float32):
        super(ModelingLayer, self).__init__(is_trainning, name, dtype)
        self.params = params
        self.feature_size = feature_size
        self.layer_number = layer_number
        self.keep_pro = 1.0 if not self.trainable else 1.0-self.params.drop_out_rate
        self.output_feature_size = self.params.rnn_hidden_size*10

    def get_output_feature_size(self):
        return self.output_feature_size

    def build(self, _):
        self.cells = [[tf.nn.rnn_cell.LSTMCell(
            self.params.rnn_hidden_size
            ,initializer=xavier_initializer()
            ,name = "fw_cell_{}".format(i)
        ),
        tf.nn.rnn_cell.LSTMCell(
            self.params.rnn_hidden_size
            ,initializer=xavier_initializer()
            ,name = "bw_cell_{}".format(i)
        )]
        for i in range(self.layer_number)]

        for i in range(self.layer_number):
            self.cells[i][0] = tf.nn.rnn_cell.DropoutWrapper( self.cells[i][0],input_keep_prob=self.keep_pro)
            self.cells[i][1] = tf.nn.rnn_cell.DropoutWrapper(self.cells[i][1],input_keep_prob=self.keep_pro)

        self.built = True

    def call(self, inputs, **kwargs):
        context = inputs[self.params.context_name]
        #M = context
        M = tf.reshape(context,[-1,self.params.c_seq_len,self.feature_size])

        context_len = inputs["context_length"]
        for i in range(self.layer_number):
            output,_ = tf.nn.bidirectional_dynamic_rnn(self.cells[i][0]
                                                           ,self.cells[i][1]
                                                           ,M
                                                           ,sequence_length=context_len
                                                           ,dtype=tf.float32
                                                       ,scope="modelling_{}".format(i))
            M = tf.concat(output,-1)
        #M =  tf.reshape(M,[-1,self.params.sent_number,self.params.c_seq_len,2*self.params.rnn_hidden_size])
        #[batch_size*sent_number,c_seq_len,2*self.params.rnn_hidden_size]
        inputs["M"] = M
        return inputs