import tensorflow as tf
from tensorflow.python.ops.lookup_ops import index_table_from_file
from common.networks.high_way import HighwayLayer
from common.networks.word_embedding import CommonWordEmbedingLayer
from common.word2vec import word_embedding_initializer
from comprehension.parameter import user_params
'''
  * Created by linsu on 2018/8/16.
  * mailto: lsishere2002@hotmail.com
'''
class WordEmbedLayer(tf.layers.Layer):
    def __init__(self, params:user_params,is_trainning=False, dtype=tf.float32, name="word_embedding"):
        super(WordEmbedLayer, self).__init__(is_trainning, name, dtype)
        self.params = params
        self.char_filters = params.char_filters
        self.feature_size = self.params.embedding_size \
            if not self.params.use_char_embedding else self.params.embedding_size+self.char_filters

    def get_output_feature_size(self):
        return self.feature_size

    def build(self, _):
        self.inner_layer = CommonWordEmbedingLayer(
            self.params.feature_voc_file_len
            ,self.params.feature_voc_file
            ,self.params.embedding_size
            ,self.params.embedding_file
            ,use_char_embedding=self.params.use_char_embedding
            ,char_embeddding_size=self.params.char_embedding_size
            ,char_conv_kernel_size=5
            ,char_filters=self.char_filters
            ,highway=True
            ,dropout_rate = self.params.drop_out_rate
            ,is_trainning = self.trainable
        )
        self.built = True


    def call(self, inputs, **kwargs):

        context = inputs[self.params.context_name]
        question = inputs[self.params.question_name]


        # #[batch_size,sent_number,seq_len]
        # context_ids = self.feature_lookup_table.lookup(inputs[self.params.context_name])
        # #[batch_size,seq_len]
        # question_ids = self.feature_lookup_table.lookup(inputs[self.params.question_name])
        # if tf.executing_eagerly():
        #     print("sentences_ids\r\n {} \r\n {}".format(context_ids,question_ids))
        # if isinstance(context_ids, tf.SparseTensor):
        #     context_ids = tf.sparse_tensor_to_dense(context_ids,default_value=0,name = "sparseid2dense_C")
        # if isinstance(question_ids,tf.SparseTensor):
        #     question_ids = tf.sparse_tensor_to_dense(question_ids,0,name = "sparseid2dense_Q")
        #
        #
        #
        # context_ids = tf.reshape(context_ids,[self.params.cur_batch_size*self.params.sent_number,self.params.c_seq_len])
        # #[batch_size*sent_number,c_seq_len], every sentence real length
        # inputs["context_mask"] = tf.abs(tf.sign(context_ids))
        # #[batch_size*sent_number]
        # sen_len_context = tf.reduce_sum(inputs["context_mask"], -1)
        # #[batch_size,q_seq_len]
        # inputs["question_mask"] = tf.abs(tf.sign(question_ids))
        # #[batch_size]
        # sen_len_question = tf.reduce_sum(inputs["question_mask"],-1)
        #
        # inputs["context_length"] = sen_len_context
        # inputs["question_length"] = sen_len_question
        # #[batch_size*sent_number,c_seq_len,1]
        # inputs["context_mask"] = tf.cast(tf.expand_dims(inputs["context_mask"],-1),tf.float32)
        # #[batch_size,q_seq_len,1]
        # inputs["question_mask"] = tf.cast(tf.expand_dims(inputs["question_mask"],-1),tf.float32)
        #
        #
        #
        # #[batch_size*sent_number,seq_len,embedding_size]
        # context_embedding = tf.nn.embedding_lookup(self.embedding,context_ids)
        # #[batch_size,seq_len,embedding_size]
        # question_embedding = tf.nn.embedding_lookup(self.embedding,question_ids)
        #
        # context_embedding = self.highway(context_embedding,mask = inputs["context_mask"])
        # question_embedding = self.highway(question_embedding,mask = inputs["question_mask"])

        context_embedding,context_seq_len,context_mask ,cxt_ids= self.inner_layer(context, chars = inputs.get(self.params.char_feature_name))
        shape = tf.shape(cxt_ids)
        order = cxt_ids._rank()
        self.params.cur_batch_size = shape[0]
        self.params.sent_number= shape[1] if order==3 else 1
        self.params.c_seq_len = shape[-1]
        self.params.q_seq_len = tf.shape(question)[-1]
        question_embedding,question_seq_len,question_mask ,qus_ids= self.inner_layer(question, chars = inputs.get(self.params.char_question_name))

        #add embeding layer dropout
        inputs[self.params.context_name] =tf.layers.dropout( tf.reshape(context_embedding,[-1,self.params.c_seq_len,self.feature_size]),self.params.drop_out_rate,training=self.trainable)
        inputs[self.params.question_name] = tf.layers.dropout(tf.reshape(question_embedding,[-1,self.params.q_seq_len,self.feature_size]),self.params.drop_out_rate,training=self.trainable)
        inputs["context_length"] = tf.reshape(context_seq_len,[-1])
        inputs["question_length"] = question_seq_len
        inputs["context_mask"] = tf.reshape(context_mask,[-1,self.params.c_seq_len,1])
        inputs["question_mask"] = question_mask
        return inputs


