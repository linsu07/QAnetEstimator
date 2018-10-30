#!-*-coding=utf-8-*-
import collections
import tensorflow as tf
import os
import numpy as np
from common.vocab_file import get_vocab_file_size


class user_params(collections.namedtuple("namedtuple",
                                         ["procedure"
                                         ,"label_name","learning_rate"
                                         ,"p1","p2"
                                         ,"embed_size",
                                          "embedding_file_path",
                                          "context_name","question_name",
                                          'rnn_hidden_size',
                                          "data_dir", "model_dir", "batch_size",
                                           "drop_out_rate","feature_voc_file_path","gpu_cores_list"
                                          ,"transfromer_conv_layers","transfromer_conv_kernel_size"
                                          ,"transfromer_head_number","tansformer_d_model"
                                          ,"clip_norm"
                                          ,"use_char_embedding"
                                          ,"char_embedding_size"
                                          ,"char_feature_name"
                                          ,"char_question_name"
                                          ,"enable_ema"
                                          ,"example_max_length"
                                          ,"char_filters"
                                          ,"ema_decay"
                                          ,"ans_limit"])):
    pass

def enrich_hyper_parameters(params: user_params):

    # feature
    if params.embedding_file_path:
        suggest_file_name = os.path.join(params.embedding_file_path, "vocabulary")
    else:
        suggest_file_name = os.path.join(params.feature_voc_file_path, "vocabulary")
    if not tf.gfile.Exists(suggest_file_name):
        files = tf.gfile.ListDirectory(params.feature_voc_file_path)
        vocab_files = [file for file in files if file.endswith(".txt")]
        vocab_file = os.path.join(params.feature_voc_file_path, vocab_files[0])
        tf.gfile.Copy(vocab_file, suggest_file_name)
    params.feature_voc_file = suggest_file_name
    params.feature_voc_file_len = get_vocab_file_size(suggest_file_name)


    # embedding
    if params.embedding_file_path:
        params.embedding_file = os.path.join(params.embedding_file_path, "embedding")
        np_array = np.loadtxt(tf.gfile.GFile(params.embedding_file, "r"))
        params.embedding_size = np_array.shape[1]
    else:
        params.embedding_size = params.embed_size
