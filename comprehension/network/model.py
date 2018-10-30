import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops.lookup_ops import index_table_from_tensor,index_table_from_file
from tensorflow.python.training import training_util

from comprehension.network.QAnet_contextual_embedding import QAnetEmbedding
from comprehension.network.QAnet_modelling_blocks import QaModelBlock
from comprehension.network.QAnet_output import QAOutputLayer
from comprehension.network.bidaf import BiDafLayer
from comprehension.network.contextual_embedding import BiLstmLayer
from comprehension.network.dcn import DcnLayer
from comprehension.network.head import SpanMatchHead
from comprehension.network.modeling import  ModelingLayer
from comprehension.network.output import OutputLayer
from comprehension.network.output_att_m2 import OutputLayerAttM2
from comprehension.network.word_embedding import WordEmbedLayer
from comprehension.parameter import user_params
from tensorflow.contrib.eager.python.tfe import py_func
'''
  * Created by linsu on 2018/8/15.
  * mailto: lsishere2002@hotmail.com
'''

def model_fn(features,labels,mode:tf.estimator.ModeKeys,config: RunConfig, params:user_params):
    print("--- model_fn in %s ---" % mode)
    num_ps_replicas = config.num_ps_replicas if config else 0
    if tf.executing_eagerly():
        partitioner = None
    else:
        partitioner = partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas)
    partitioner = None
    def is_training():
        return mode == tf.estimator.ModeKeys.TRAIN
    layers = list()
    with tf.variable_scope("rc", partitioner=partitioner, initializer=xavier_initializer()):
        feature_size = 0
        #params.procedure = ["wordembedding,contextual,bidaf,modeling,output"]
        for name in params.procedure:
            if name == "wordembedding":
                l=WordEmbedLayer(params,is_training())
            elif name == "contextual":
                l=(BiLstmLayer(params,feature_size,is_training()))
            elif name == "bidaf":
                l=(BiDafLayer(params,feature_size,is_training()))
            elif name == "dcn":
                l=(DcnLayer(params,feature_size,is_training()))
            elif name == "modeling":
                l=(ModelingLayer(params,feature_size,2,is_training(),name = "m1"))
            elif name == "output":
                l=(OutputLayer(params,feature_size,is_training()))
            elif name == "output_att":
                l=(OutputLayerAttM2(params,feature_size,is_training()))
            elif name == "QAnetEmbedding":
                l=(QAnetEmbedding(params,feature_size,is_training()))
            elif name == "QaModelBlock":
                l=(QaModelBlock(params,feature_size,is_training()))
            elif name=="QAOutput":
                l=QAOutputLayer(params,feature_size,is_training())
            else:
                raise ValueError("unknow precedure, valid name is pcnn,mi_att,birnn")
            layers.append(l)
            feature_size = l.get_output_feature_size()

        head = SpanMatchHead(name="span_head")
        logits = features
        for layer in layers:
            logits = layer(logits)

    def train_op_fn(loss):
        global_step=training_util.get_global_step()
        #warmed up learning rate
        lr = tf.minimum(params.learning_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(global_step, tf.float32) + 1))
        opt = tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)
        grads = opt.compute_gradients(loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(
            gradients, params.clip_norm)
        train_op = opt.apply_gradients(
            zip(capped_grads, variables), global_step=global_step)
        return train_op
        # return tf.train.AdamOptimizer(learning_rate=params.learning_rate) \
        #     .minimize(loss, global_step=training_util.get_global_step())

    if params.enable_ema and mode==tf.estimator.ModeKeys.TRAIN:
        ema = tf.train.ExponentialMovingAverage(decay=params.ema_decay)
        trained_var = tf.trainable_variables()
        ema_op = ema.apply(trained_var)
        # for var in tf.get_collection(key = "not_in_ema"):
        #     trained_var.remove(var)
        variables_to_restore = ema.variables_to_restore()
        #print( "in train-------------------------------------")
        #print(variables_to_restore)
        with tf.control_dependencies([ema_op]):
            logits = tf.identity(logits[0]),tf.identity(logits[1])

    #regularization_loss = None
    lamda  = 3e-7
    regularization_loss = [lamda * x for x in tf.losses.get_regularization_losses()]
    ps  = None
    if mode==tf.estimator.ModeKeys.TRAIN or mode==tf.estimator.ModeKeys.EVAL:
        ps = [labels[params.p1],labels[params.p2]]

    spec = head.create_estimator_spec(
        None, mode, logits, labels=ps, train_op_fn=train_op_fn, regularization_losses=regularization_loss,params= params)
    if params.enable_ema and mode==tf.estimator.ModeKeys.PREDICT:
        ema = tf.train.ExponentialMovingAverage(decay=params.ema_decay)
        variables_to_restore = ema.variables_to_restore()
        #print( "in prodict-------------------------------------")
        #print(variables_to_restore)
        scaffold = spec.scaffold
        scaffold._saver =tf.train.Saver(variables_to_restore)

    return spec


