import tensorflow as tf
from tensorflow.python.ops.lookup_ops import index_table_from_file, index_table_from_tensor
from common.networks.high_way import HighwayLayer
from common.networks.word2vec import word_embedding_initializer
'''
  * Created by linsu on 2018/9/29.
  * mailto: lsishere2002@hotmail.com
'''
class CommonWordEmbedingLayer(tf.layers.Layer):
    def __init__(self, voc_size, voc_file, embedding_size, embedding_file = None
                 ,use_char_embedding = False,char_embeddding_size = 16,char_filters = 16,char_conv_kernel_size=5
                 , highway = False,dropout_rate = 0.0, is_trainning = False, dtype=tf.float32, name="word_embedding"):
        super(CommonWordEmbedingLayer, self).__init__(is_trainning, name, dtype)
        self.num_oov_buckets = 1
        self.voc_size = voc_size
        self.embedding_size = embedding_size
        self.embedding_file = embedding_file
        self.voc_file = voc_file
        self.use_highway = highway
        self.use_char_embedding = use_char_embedding
        self.char_filters = char_filters
        self.char_embeddding_size = char_embeddding_size
        self.char_conv_kernel_size = char_conv_kernel_size
        self.dropout_rate = dropout_rate

    def build(self, _):
        self.word_embedding = self.add_variable(
            name="word_embedding"
            , shape=[1+self.voc_size + self.num_oov_buckets, self.embedding_size] #1 for padding
            , initializer=word_embedding_initializer(
                self.embedding_file, include_word=False) if self.embedding_file!=None else tf.random_uniform_initializer(-1, 1)
            ,regularizer=tf.nn.l2_loss
        )
        self.feature_lookup_table = index_table_from_file(
            vocabulary_file=self.voc_file,
            num_oov_buckets=self.num_oov_buckets,
            vocab_size=self.voc_size,
            default_value=-1,
            key_dtype=tf.string,
            name='feature_index_lookup')

        if self.use_char_embedding:
            char_list = ["0","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",".","'"]
            char_list = tf.constant(char_list,dtype=tf.string)
            self.char_lookup_table = index_table_from_tensor(char_list,100,default_value=-1)# default_value must be -1
            self.char_embeddding = self.add_variable("char_embedding",[27+2+100,self.char_embeddding_size]
                                                     ,initializer=tf.random_uniform_initializer(-1, 1)
                                                     ,regularizer=tf.nn.l2_loss)
            self.charConv = tf.layers.Conv1D(
                filters= self.char_filters,
                kernel_size=self.char_conv_kernel_size
                ,activation=tf.nn.relu
                ,use_bias=True
                ,bias_initializer=tf.zeros_initializer()
                ,trainable=True
                ,kernel_regularizer=tf.nn.l2_loss
            )

        if self.use_highway:
            self.highway = HighwayLayer(self.embedding_size+self.char_filters if self.use_char_embedding else self.embedding_size
                                        ,layers_number=2,is_trainning=self.trainable,dropout_rate=self.dropout_rate)

        self.built = True
    """
            return {
                "context":context_embedding
                ,"context_len":sen_len
                ,"context_mask":mask
            }
    """
    def call(self, words, **kwargs):
        #e.g [batch_size,sent_number,seq_len]
        context_ids = self.feature_lookup_table.lookup(words)
        if isinstance(context_ids, tf.SparseTensor):
            context_ids = tf.sparse_tensor_to_dense(context_ids,default_value=0,name = "sparseid2dense_C")
        #[batch_size,sent_number,c_seq_len], every sentence real length
        mask = tf.abs(tf.sign(context_ids))
        #[batch_size, sent_number]
        sen_len = tf.reduce_sum(mask, -1)
        #[batch_size,sent_number,c_seq_len,1],
        mask = tf.expand_dims(tf.cast(mask,tf.float32),-1)
        context_embedding = tf.nn.embedding_lookup(self.word_embedding,context_ids)
        context_embedding = tf.layers.dropout(context_embedding,self.dropout_rate,training=self.trainable)

        if self.use_char_embedding :
            chars = kwargs.get("chars")
            shape = tf.shape(chars)
            if isinstance(chars, tf.SparseTensor):
                chars= tf.sparse_reshape(chars,[-1])
                chars = tf.sparse_tensor_to_dense(chars,default_value="0")
            else:
                chars = tf.reshape(chars,[-1])
            chars = tf.string_split(chars,delimiter=",")
            chars_ids = self.char_lookup_table.lookup(chars)
            chars_ids = tf.sparse_tensor_to_dense(chars_ids)
            char_mask = tf.cast(tf.expand_dims(tf.sign(chars_ids),-1),tf.float32)
            char_embedding = tf.nn.embedding_lookup(self.char_embeddding,chars_ids)*char_mask #context_mask cares about this
            char_embedding = tf.layers.dropout(char_embedding,self.dropout_rate*0.5,training=self.trainable)
            char_embedding = self.conv_chars(char_embedding,shape)
            context_embedding = tf.concat([context_embedding,char_embedding],-1)
        context_embedding = self.highway(context_embedding,mask = mask) if self.use_highway else context_embedding*mask

        return context_embedding,sen_len,mask,context_ids


    def conv_chars(self,char_embedding,shape):
        #shape = tf.shape(char_embedding)
        #char_embedding = tf.reshape(char_embedding,[-1,seq,self.char_embeddding_size])
        ret = self.charConv(char_embedding)
        ret = tf.reduce_max(ret,-2)
        ret = tf.reshape(ret,tf.concat([shape,[self.char_filters]],axis = 0))
        return ret

