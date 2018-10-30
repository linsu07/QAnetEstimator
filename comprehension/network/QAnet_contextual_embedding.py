import tensorflow as tf

from common.networks.attention_is_all import GoogleTransformer
from comprehension.parameter import user_params

class QAnetEmbedding(tf.layers.Layer):
    def __init__(self, params:user_params,feature_size:int, is_trainning=False, name="QAnetEmbedding", dtype=tf.float32):
        super(QAnetEmbedding, self).__init__(is_trainning, name, dtype)
        self.params = params
        self.feature_size = feature_size
        self.output_feature_size = self.params.tansformer_d_model

    def get_output_feature_size(self):
        return self.output_feature_size

    def build(self, _):
        self.transformer = GoogleTransformer(
            head_number= self.params.transfromer_head_number,
            orginal_feature_size=self.feature_size,
            d_model=self.params.tansformer_d_model
            ,is_trainning=self.trainable
            ,name = "qanet_encoding_transformer"
            ,deep_wise_cnn=True
            ,cnn_layers_count=4
            ,cnn_kernel_size=7
            ,positionAware=True,dropout_rate=self.params.drop_out_rate
        )
        self.built = True
    def call(self, inputs, **kwargs):

        #for context
        #[batch_size*sent_num,seq_len,feature_size]
        feature = inputs[self.params.context_name]
        # shape = tf.shape(feature)
        # batch_size,sent_num,seq_len,feature_size = shape[0],shape[1],shape[2],shape[3]
        # feature = tf.reshape(feature,[batch_size*sent_num,seq_len,feature_size])
        #mask = tf.reshape(inputs["context_mask"],[batch_size*sent_num,seq_len])
        #[batch_size*sent_num,seq_len]
        mask = inputs["context_mask"]
        feature = self.transformer(feature,mask=mask)
        #inputs[self.params.context_name] = tf.reshape(feature,[batch_size,sent_num,seq_len,self.params.tansformer_d_model])
        #clean
        inputs[self.params.context_name] = feature

        #for question
        feature = inputs[self.params.question_name]
        mask = inputs["question_mask"]
        #clean
        inputs[self.params.question_name] = self.transformer(feature,mask=mask)

        return inputs
