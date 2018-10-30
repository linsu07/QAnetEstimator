import math

import six
import tensorflow as tf
from common.networks.deep_wise_cnn import DeepWiseCnn
from common.networks.position_wise import PositionAwareLayer

'''
  * Created by linsu on 2018/8/10.
  * mailto: lsishere2002@hotmail.com
'''
from tensorflow.contrib.layers import xavier_initializer


class GoogleTransformer(tf.layers.Layer):
    def __init__(self
                 ,head_number = 8
                 ,orginal_feature_size = 128
                 ,d_model = 128
                 ,is_trainning=False
                 , name="GoogleTransformer"
                 ,deep_wise_cnn = False
                 ,cnn_layers_count = 4
                 ,cnn_kernel_size = 5
                 ,positionAware = False
                 ,dropout_rate = 0.1
                 , dtype=tf.float32):
        super(GoogleTransformer, self).__init__(is_trainning, name, dtype)
        self.d_kv = d_model/head_number
        self.orginal_feature_size= orginal_feature_size
        self.d_model = d_model
        self.use_cnn = deep_wise_cnn
        self.cnn_layers_count = cnn_layers_count
        self.cnn_kernel_size = cnn_kernel_size
        self.head_number = head_number
        self.positionAware = positionAware
        self.dropout_rate = dropout_rate
        self.scale =1/ math.sqrt(self.d_kv)
        self.exp_epsilon = -1.0e20
        self.epsilon = 1e-6

    def build(self, feature_shape):

        # size =feature_shape[-1]
        self.raw_prj_weight = self.add_variable("raw_prj_weight"
                                                ,shape = [self.orginal_feature_size,self.d_model]
                                                ,regularizer=tf.nn.l2_loss)
        self.raw_prj_bias = self.add_variable("raw_prj_bias",shape=[self.d_model],initializer=tf.zeros_initializer())
        self.query_prj =tf.layers.Dense(
            self.d_model,
            kernel_initializer=xavier_initializer(),
            kernel_regularizer=tf.nn.l2_loss,
            use_bias=True
            ,name = "query_proj"
        )
        self.key_prj = tf.layers.Dense(
            self.d_model
            ,
            kernel_initializer=xavier_initializer(),
            kernel_regularizer=tf.nn.l2_loss
            ,use_bias=True
            ,name = "key_prj" )
        self.value_prj = tf.layers.Dense(
            self.d_model,
            kernel_initializer=xavier_initializer(),
            kernel_regularizer=tf.nn.l2_loss
            ,use_bias=True
            ,name = "value_prj")

        self.feature_prj = tf.layers.Dense(
            self.d_model,
            kernel_initializer=xavier_initializer(),
            kernel_regularizer=tf.nn.l2_loss
            ,name = "feature_prj")

        # self.beta = self.add_variable("beta1",shape = [self.d_model],initializer=tf.zeros_initializer())
        # self.gamma = self.add_variable("gamma1",shape = [self.d_model],initializer=tf.ones_initializer())
        #
        #
        # self.beta2 = self.add_variable("beta2",shape = [self.d_model],initializer=tf.zeros_initializer())
        # self.gamma2 = self.add_variable("gamma2",shape = [self.d_model],initializer=tf.ones_initializer())

        self.ffn_prj_1 = tf.layers.Dense(
            self.d_model,
            kernel_initializer=xavier_initializer(),
            kernel_regularizer=tf.nn.l2_loss
            ,name = "ffn_prj_1"
            ,activation=tf.nn.relu
        )

        self.ffn_prj_2 = tf.layers.Dense(
            self.d_model,
            kernel_initializer=xavier_initializer(),
            kernel_regularizer=tf.nn.l2_loss
            ,name = "ffn_prj_2"
        )
        self.cnn_layers = []
        if self.use_cnn:
            for i in range(self.cnn_layers_count):
                self.cnn_layers.append(DeepWiseCnn(self.cnn_kernel_size
                                                   ,self.d_model,self.trainable,self.dropout_rate
                                                   ,name = "deep_wise_cnn_{}".format(i)))
        self.built = True


    def expand_size_and_reshape(self,feature,mask):
        if(self.orginal_feature_size==self.d_model):
            return feature
        seq_len= tf.shape(feature)[1]
        feature = tf.reshape(feature,[-1,self.orginal_feature_size],name="raw_reshape_1")
        feature = tf.matmul(feature,self.raw_prj_weight,name="raw_reshpae_2")+self.raw_prj_bias
        feature = tf.reshape(feature,[-1,seq_len,self.d_model],name="raw_reshape_3")
        if mask!=None:
            #mask = tf.expand_dims(mask,-1)
            feature = tf.multiply(feature,mask)
        return feature


    def deep_wise_cnn(self,feature,mask):
        # if mask!=None:
        #     feature = feature*mask
        feature = tf.expand_dims(feature,axis=2)
        mask = tf.expand_dims(mask,axis=2)
        # feature = feature*mask
        for i,cnn_layer in enumerate(self.cnn_layers):
            #dirty for padding
            orig = feature
            layer_dropout_rate = (i/(self.cnn_layers_count))*(self.dropout_rate)
            random = tf.random_uniform([],minval=0,maxval=1)

            #dirty for padding
            feature = self.layer_norm(feature,name = "deep_wise_{}".format(i))
            feature = feature*mask  #clean
            if i%2==0:
                feature = tf.layers.dropout(feature,self.dropout_rate,training=self.trainable)
            #dirty for padding
            output =cnn_layer(feature)
            #dirty for padding
            #layer dropout
            feature = tf.cond(tf.less(random,layer_dropout_rate),
                              lambda :orig,lambda :output+orig) if self.trainable else output+orig

        feature = feature*mask  #clean
        feature = tf.squeeze(feature,axis=2)
        return feature
    """
    feature shape must be [batch_size(maybe include sent_num),seq_len,feature_size]
    mask must be [batch_size(maybe include sent_num),seq_len]；1 or o inside
    """
    def call(self, inputs, **kwargs):
        mask = kwargs.get('mask')
        feature = self.expand_size_and_reshape(inputs,None)#,mask=mask) #dirty for padding
        if self.positionAware:
            pos_maker = PositionAwareLayer()
            feature = pos_maker(feature)#,mask=mask) #dirty for padding
        if self.use_cnn:
            feature = self.deep_wise_cnn(feature,mask) #mask clean out

        #clean
        attention_feature,mask_expand_last = self.mutilhead_attention(feature,feature,mask)
        feature = self.ffn(attention_feature,mask_expand_last)
        #feature = tf.reshape(feature,[-1,self.sent_num,self.seq_len,self.d_model])
        # inputs[self.params.feature_name] = feature
        #clean
        return feature

    def mutilhead_attention(self,query_raw,key_value,mask):

        #dirty for padding
        query = self.layer_norm(query_raw,name = "multi_head")
        query = tf.layers.dropout(query,self.dropout_rate,training=self.trainable)
        key_value = query
        #query :[batch,t_q,d_model]
        #key,value : [batch,t_k,d_model]
        #dirty for padding
        query = self.query_prj(query)
        #[batch*head_number,t_q,d_kv]
        query = tf.concat(tf.split(query,self.head_number,axis = -1,name="query_split_5"),axis=0,name="query_concat_6")
        key = self.key_prj(key_value)
        #[batch*head_number,t_k,d_kv]
        key = tf.concat(tf.split(key,self.head_number,axis = -1,name="key_split_7"),axis=0,name = "key_concat_8")
        #[batch*head_number,t_q,t_k]
        factors_raw = tf.multiply(tf.matmul(query,key,transpose_b=True,name="factor_matmul_9"),self.scale,name = "facotor_mul_10")

        mask_expand_last = mask
        mask_expand_before = tf.expand_dims(tf.squeeze(mask,-1),-2)
        #[batch*head_number,t_q,t_k]
        softmax_padding = tf.tile((1.0- tf.multiply(mask_expand_last,mask_expand_before))*self.exp_epsilon,multiples=[self.head_number,1,1])

        factors = factors_raw+softmax_padding # minus number small enough , need not to *mask

        #factors_mask = tf.sign(tf.abs(factors_raw,name="factor_abc_11"),name="facotr_sign_12")
        #factors =  tf.where(tf.cast(factors_mask,tf.bool,name = "facotor_cast_13")
                            # ,factors_raw,tf.multiply(tf.ones_like(factors_raw),self.exp_epsilon,name="factor_mul_14")
                            # ,name="factor_where_15")

        factors = tf.multiply(tf.nn.softmax(factors,axis=-1,name="factor_softmax_16"),tf.tile(mask_expand_last,multiples=[self.head_number,1,1]))
        #factors = tf.multiply(factors,factors_mask,name = "factor_mul_17")

        #dirty for padding
        value = self.value_prj(key_value)
        #[batch*head_number,t_k,d_kv]
        value = tf.concat(tf.split(value,self.head_number,axis = -1,name = "value_split_18"),axis=0,name="value_concat_19")
        #[batch*head_number,t_q,d_kv]  #clean :)
        query_value = tf.matmul(factors,value,name="value_matmul_20")
        #[batch,t_q,d_model]
        feature = tf.concat(tf.split(query_value,num_or_size_splits=self.head_number,axis=0,name="value_split_21"),axis=-1,name="value_concat_22")
        feature = self.feature_prj(feature)#dirty for padding
        feature = feature+ query_raw

        feature = tf.multiply(feature,mask_expand_last)#clean
        # feature = self.layer_norm(feature,1)
        return feature,mask_expand_last

    def layer_norm(self,inputs ,name,**kwargs):
        with tf.variable_scope(name):
            # mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True,name="moments_24")
            mean = tf.reduce_mean(inputs, axis=[-1], keep_dims=True)
            variance = tf.reduce_mean(tf.square(inputs - mean), axis=[-1], keep_dims=True)
            gamma = tf.get_variable(shape=[self.d_model],initializer=tf.ones_initializer(), name="gamma",regularizer=tf.nn.l2_loss)
            beta = tf.get_variable(shape = [self.d_model],initializer=tf.zeros_initializer(),name = "beta",regularizer=tf.nn.l2_loss)
            #gamma = tf.constant(1.0, shape=[self.d_model], dtype=tf.float32)
            normalized = tf.divide((inputs - mean) ,  (variance + self.epsilon) ** (.5),name = "norm_divide_25" )
            outputs = gamma * normalized + beta
        return outputs

    def ffn(self,input,mask):
        #dirty
        feature = self.layer_norm(input,name = "ffn_norm")
        feature = tf.layers.dropout(feature,self.dropout_rate,training=self.trainable)
        middle_result = self.ffn_prj_1(feature)
        feature = self.ffn_prj_2(middle_result)
        #feature = tf.layers.dropout(feature,self.dropout_rate,training=self.trainable)
        feature = tf.add(input , feature,name = "ffn_add_26")
        feature = tf.multiply(feature,mask)

        return feature


