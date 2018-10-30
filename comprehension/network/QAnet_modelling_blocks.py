import tensorflow as tf

from common.networks.attention_is_all import GoogleTransformer
from comprehension.parameter import user_params


class QaModelBlock(tf.layers.Layer):
    def __init__(self, params:user_params,feature_size:int, is_trainning=False, name="QaModelBlock", dtype=tf.float32):
        super(QaModelBlock, self).__init__(is_trainning, name, dtype)
        self.params = params
        self.feature_size = feature_size
        self.output_feature_size = self.params.tansformer_d_model

    def get_output_feature_size(self):
        return self.output_feature_size

    def build(self, _):
        blocks = 7
        self.block_list = []
        ori_size = self.feature_size
        for i in range(blocks):
            block =  GoogleTransformer(
                head_number= self.params.transfromer_head_number
                ,orginal_feature_size=ori_size
                ,d_model=self.params.tansformer_d_model
                ,is_trainning=self.trainable,name = "qanet_modeling_transformer_{}".format(i)
                ,deep_wise_cnn=True
                ,cnn_layers_count=2
                ,cnn_kernel_size=5
                ,positionAware=True,dropout_rate=self.params.drop_out_rate
            )
            ori_size = self.params.tansformer_d_model
            self.block_list.append(block)

    def call(self, inputs, **kwargs):
        #for context
        #[batch_size*sent_num,seq_len,feature_size]
        feature = inputs[self.params.context_name]
        # shape = tf.shape(feature)
        # batch_size,sent_num,seq_len,feature_size = shape[0],shape[1],shape[2],shape[3]
        # feature = tf.reshape(feature,[batch_size*sent_num,seq_len,feature_size])
        # ,[batch_size*sent_num,seq_len]
        mask = inputs["context_mask"]

        inputs["M0"] = self.modeling(feature,mask=mask)
        self.block_list[0].orginal_feature_size = self.params.tansformer_d_model
        inputs["M1"] = self.modeling(inputs["M0"] ,mask=mask)
        inputs["M2"] = self.modeling(inputs["M1"],mask= mask)

        # inputs["M0"] = tf.reshape(inputs["M0"],[batch_size,sent_num,seq_len,self.params.tansformer_d_model])
        # inputs["M1"] = tf.reshape(inputs["M1"],[batch_size,sent_num,seq_len,self.params.tansformer_d_model])
        # inputs["M2"] = tf.reshape(inputs["M2"],[batch_size,sent_num,seq_len,self.params.tansformer_d_model])

        return inputs



    def modeling(self, feature, mask):
        for block in self.block_list:
            feature = block(feature,mask=mask)
        return feature