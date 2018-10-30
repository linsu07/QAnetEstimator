import tensorflow as tf
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
import os
from tensorflow.python.lib.io import file_io
from comprehension.parameter import user_params
'''
  * Created by muyongli on 2018/8/29.
'''
class SparkInput:
    def __init__(self,params:user_params):
        self.context_name= params.context_name
        self.p1 = params.p1
        self.p2 = params.p2
        self.question_name=params.question_name

    def get_data_dir(self, mode: tf.estimator.ModeKeys, root: str):
        return os.path.join(root, "train" if (mode == tf.estimator.ModeKeys.TRAIN) else "evaluation")

    def input_fn(self, mode: tf.estimator.ModeKeys, params: user_params,data_dir:str):
        gpu_cores = 1 if not params.gpu_cores_list else len(params.gpu_cores_list)

        conttext_spec ={
             params.question_name: tf.VarLenFeature(tf.string),
            "context_len":tf.FixedLenFeature([],tf.int64,0)
        }
        sequence_spec = {
            params.p1:tf.VarLenFeature(tf.int64),
            params.p2:tf.VarLenFeature(tf.int64),
            params.context_name:tf.VarLenFeature(tf.string)
        }
        if params.use_char_embedding:
            sequence_spec[params.char_feature_name] = tf.VarLenFeature(tf.string)
            conttext_spec[params.char_question_name] =  tf.VarLenFeature(tf.string)
        file_paths = file_io.get_matching_files(os.path.join(data_dir, "part-r-*"))#返回一个列表
        data_set = tf.data.TFRecordDataset(file_paths, buffer_size=80 * 1024)
        batch_size = gpu_cores * params.batch_size
        print("*******cur batch_size is {}".format(batch_size))
        def parse(raw):

            array_String_dic,array_array_String_dic =tf.parse_single_sequence_example(serialized=raw,context_features=conttext_spec
                                                                  ,sequence_features=sequence_spec)
            context  = array_array_String_dic.get(self.context_name)
            question = array_String_dic.get(self.question_name)
            context_len = array_String_dic.get("context_len")
            p1_labels=array_array_String_dic.get(self.p1)
            p2_labels=array_array_String_dic.get(self.p2)
            if params.use_char_embedding:
                char_context = array_array_String_dic.get(params.char_feature_name)
                char_question = array_String_dic.get(params.char_question_name)
                return { params.context_name: context,  params.question_name:question,"context_len":context_len
                           ,params.char_feature_name:char_context,params.char_question_name:char_question} \
                    ,{ params.p1:p1_labels,   params.p2:p2_labels}
            else:
                return { params.context_name: context,  params.question_name:question,"context_len":context_len}\
                ,{ params.p1:p1_labels,   params.p2:p2_labels}

        def predicate_func(feature,label):
            data_size = feature.get("context_len")
            return tf.less(data_size,params.example_max_length)

        if mode == tf.estimator.ModeKeys.TRAIN:
            data_set = data_set.repeat(None).shuffle(buffer_size=1000 *1024) \
                .map(parse).filter(predicate_func).batch(batch_size)#.prefetch(buffer_size=None)
        elif mode == tf.estimator.ModeKeys.EVAL:
            data_set = data_set.repeat(None) \
                .take(8000).map(parse).batch(60)#.prefetch(buffer_size=None)
        return data_set

    def get_input_reciever_fn(self):
        context = tf.placeholder(dtype=tf.string, shape=[None, None, None], name="context_tensor")
        question = tf.placeholder(dtype=tf.string, shape=[None, None], name = "query_tensor")
        receiver_tensors = { self.context_name: context,self.question_name:question }
        return build_raw_serving_input_receiver_fn(receiver_tensors)
