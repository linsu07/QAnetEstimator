#!-*-coding=utf-8-*-
import tensorflow as tf
import sys
import platform
system = str(platform.platform()).lower()
#print("system is: " + system)
if system.startswith('windows'):
    sys.path.insert(0, 'D:/program/program4/dev-nlp/ultra-nlp-tensorflow/src')
if system.startswith('linux'):
    sys.path.insert(0, '/usr/local/python3/lib/python3.6/site-packages')

#sys.path.insert(0, "D:/program/program4/dev-nlp/ultra-nlp-tensorflow/src")
#sys.path.insert(0, "linux path")



from tensorflow.python.estimator.run_config import RunConfig, TaskType
from comprehension.input import SparkInput
from comprehension.network.model import model_fn
from comprehension.parameter import user_params, enrich_hyper_parameters
from common import MyTraining
from common.listeners import EvalListener, LoadEMAHook

'''
  * Created by muyongli on 2018/8/29.
'''

FLAGS=tf.app.flags.FLAGS
tf.flags.DEFINE_list("procedure",["wordembedding","contextual","bidaf","modeling","output_att"],"bidaf各个网络层的名字列表")
tf.flags.DEFINE_string("label_name", "label", "tfrecord中的标签的名字")
tf.app.flags.DEFINE_float("learning_rate", 0.001, '学习率.')
tf.app.flags.DEFINE_integer("embed_size",100,"如果使用预先训练的embedding，此参数无效，即embedding_file_path 不为None")
tf.flags.DEFINE_string("embedding_file",None,"embedding词向量文件")# voc_file 第一个词必须是<padding>, 第二个词必须是<unk>
tf.flags.DEFINE_string("embedding_file_path",None,"可选，预训练的embedding文件路径，包括embedding和vocabulary 2个文件，如果不为none，embed_size，feature_voc_file_path参数不起作用")
tf.app.flags.DEFINE_string("context_name","context","篇章内容")
tf.app.flags.DEFINE_string("question_name","question","问题")
tf.app.flags.DEFINE_integer("rnn_hidden_size",100,"rnn cell的隐藏层长度")
tf.app.flags.DEFINE_string('data_dir', 'd:\\bidaf\\tfrecod\\', '训练数据存放路径，支持hdfs')
tf.app.flags.DEFINE_string('model_dir', 'd:\\bidaf\\model\\', '保存dnn模型文件的路径，支持hdfs')
tf.app.flags.DEFINE_integer('batch_size', 20, '一批数量样本的数量')
tf.app.flags.DEFINE_float('drop_out_rate',0.3,"神经网络随机选取不输出数据到输出层的比例，这样做是为了防止过拟合和提高预测正确率")
tf.app.flags.DEFINE_string("p1","p1","答案开始标签")
tf.app.flags.DEFINE_string("p2","p2","答案结束标签")
tf.app.flags.DEFINE_integer('max_steps', 1000, '训练模型最大的批训练次数，在model_dir不变的情况下重复训练'
                                               '，达到max_step后，不再继续训练，或者增加max_step，或者更换model_dir, 再继续训练')
tf.app.flags.DEFINE_integer("check_steps", 300,'保存训练中间结果的间隔，也是evalutation的间隔')
tf.app.flags.DEFINE_string('log_level', 'INFO', 'tensorflow训练时的日志打印级别， 取值分别为，DEBUG，INFO,WARN,ERROR')
tf.app.flags.DEFINE_string("gpu_cores",None,"例如'[0,1,2,3]'，在当个GPU机器的情况，使用的哪些核来训练")
tf.app.flags.DEFINE_string("feature_voc_file_path", None, "tfrecord中的特征词的字典文件地址，为了兼容spark，目录下唯一text为file")

tf.flags.DEFINE_integer("transfromer_conv_layers",4,"使用google 的transformer的时候deepwise cnn的层数，=0意味着不启用cnn")
tf.flags.DEFINE_integer("transfromer_conv_kernel_size",5,"使用google 的transformer的时候deepwise cnn的卷积核的长度")
tf.flags.DEFINE_integer("transfromer_head_number",8,"使用google 的transformer每一层的head的数量")
tf.flags.DEFINE_integer("tansformer_d_model",128,"使用google 的transformer每一层feature的维度，最好和embedding_size相等")

tf.flags.DEFINE_float("clip_norm",5.0,"在计算梯度下降时候的clip_norm，一般不要修改")
tf.flags.DEFINE_integer("use_char_embedding",0,"是否使用对字母的卷积，只有是英文的时候才可用")
tf.flags.DEFINE_integer("char_embedding_size",16,"使用对字母的卷积时候，每个字母embedding的长度，只有是英文的时候才可用")
tf.flags.DEFINE_string("char_feature_name","chars","使用对字母的卷积时候，char列的名字，只有是英文的时候才可用")
tf.flags.DEFINE_string("char_question_name","ques_chars","使用对字母的卷积时候，char列的名字，只有是英文的时候才可用")
tf.flags.DEFINE_integer("enable_ema",0,"是否启动指数移动平均来计算参数")
tf.flags.DEFINE_integer("example_max_length",500,"训练数据最大的长度， 预测数据不限制")
tf.flags.DEFINE_integer("char_filters",100,"训练数据最大的长度， 预测数据不限制")
tf.flags.DEFINE_float("ema_decay",0.9999,"ema的decay速率")
tf.flags.DEFINE_integer("ans_limit",30,"answer 的最大长度")


#可能需要的参数

#HADOOP_USER_NAME=root CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob)
#  CUDA_VISIBLE_DEVICES=5 python3 /disk1/linsu/ultra/comprehension/train.py --learning_rate 0.001
#   --data_dir  /disk1/linsu/data/ --model_dir /disk1/linsu/model/qanet --max_steps 30000 --batch_size 20
# --check_steps 1000 --log_level INFO  --drop_out_rate 0.1 --embed_size 100
# --feature_voc_file_path hdfs://192.168.181.159:9000/linsu/rc/feature_voc_file/ --embedding_file
# --label_name label --context_name context --question_name question --transfromer_head_number 8
# --tansformer_d_model 128 --p1 p1 --p2 p2
# --procedure "wordembedding","QAnetEmbedding","dcn","QaModelBlock","QAOutput" --example_max_length 420
# --use_char_embedding 1 --char_feature_name char_context --char_question_name char_question --enable_ema 1
# --char_filters 100 --char_embedding_size 16 --ema_decay 0.9999
# suggestion from QAnet doc


def bidaf_train(_):

    params=user_params(procedure=tuple(FLAGS.procedure),
                       label_name=FLAGS.label_name,learning_rate=FLAGS.learning_rate,
                       embed_size=FLAGS.embed_size,
                       embedding_file_path=FLAGS.embedding_file_path,
                       context_name=FLAGS.context_name,question_name=FLAGS.question_name,
                       rnn_hidden_size=FLAGS.rnn_hidden_size,data_dir=FLAGS.data_dir,model_dir=FLAGS.model_dir,
                       batch_size=FLAGS.batch_size,drop_out_rate=FLAGS.drop_out_rate,p1=FLAGS.p1,
                       p2=FLAGS.p2,feature_voc_file_path=FLAGS.feature_voc_file_path,
                       gpu_cores_list=FLAGS.gpu_cores,
                       transfromer_conv_layers = FLAGS.transfromer_conv_layers,
                       transfromer_conv_kernel_size = FLAGS.transfromer_conv_kernel_size
                       ,transfromer_head_number = FLAGS.transfromer_head_number
                       ,tansformer_d_model = FLAGS.tansformer_d_model
                       ,clip_norm = FLAGS.clip_norm
                       ,use_char_embedding = FLAGS.use_char_embedding
                       ,char_embedding_size = FLAGS.char_embedding_size
                       ,char_feature_name = FLAGS.char_feature_name
                       ,char_question_name = FLAGS.char_question_name
                       ,example_max_length = FLAGS.example_max_length
                       ,enable_ema=FLAGS.enable_ema
                       ,ema_decay= FLAGS.ema_decay
                       ,char_filters = FLAGS.char_filters
                       ,ans_limit = FLAGS.ans_limit
                       )

    #词向量文件的加载
    enrich_hyper_parameters(params)

    # 配置日志等级
    level_str = 'tf.logging.{}'.format(str(tf.flags.FLAGS.log_level).upper())
    tf.logging.set_verbosity(eval(level_str))

    #加载数据,创建一个SparkInput类对象
    input = SparkInput(params)

    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess_config.gpu_options.allow_growth = True
    #sess_config.report_tensor_allocations_upon_oom = True
    #sess_config.log_device_placement = True

# estimator运行环境配置
    if FLAGS.gpu_cores:
        gpu_cors = tuple(eval(FLAGS.gpu_cores))#FLAGS.gpu_cores
        devices =  ["/device:GPU:%d" % d for d in gpu_cors]#"/device:GPU:%d" % d作为元组中的一个元素整体
        distribution = tf.contrib.distribute.MirroredStrategy(devices = devices)#distribution是一个MirroredStrategy类
        config = RunConfig(save_checkpoints_steps=FLAGS.check_steps,train_distribute=distribution)
    else:
        config = RunConfig(save_checkpoints_steps=FLAGS.check_steps,session_config=sess_config)#config是一个RunConfig类对象

    #estimator创建
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, config=config, params=params)

    #得到训练和测试数据的文件路径
    train_data_dir = input.get_data_dir(tf.estimator.ModeKeys.TRAIN, params.data_dir)
    eval_data_dir =  input.get_data_dir(tf.estimator.ModeKeys.EVAL, params.data_dir)

    #创建EvalListener进行训练和预测的评估
    hook = [] if not params.enable_ema else [LoadEMAHook(params.model_dir,FLAGS.ema_decay)]
    listeners = [
        EvalListener(estimator, lambda: input.input_fn(mode = tf.estimator.ModeKeys.EVAL, params=params,data_dir=train_data_dir), name="train_data",hook = hook),
        EvalListener(estimator, lambda: input.input_fn(mode = tf.estimator.ModeKeys.EVAL, params=params,data_dir=eval_data_dir),hook = hook)
    ]

    #由训练数据的文件路径获取训练数据
    def train_input_fn():
        return input.input_fn(mode = tf.estimator.ModeKeys.TRAIN, params=params,data_dir=train_data_dir)

    #gpu cluster
    if config.cluster_spec:
        train_spec = MyTraining.TrainSpec(train_input_fn, FLAGS.max_steps)
        eval_spec = MyTraining.EvalSpec(lambda: input.input_fn(mode = tf.estimator.ModeKeys.EVAL, params=params), steps=FLAGS.check_steps)
        MyTraining.train_and_evaluate(estimator, train_spec, eval_spec, listeners)
        if config.task_type == TaskType.CHIEF:
            model_dir = estimator.export_savedmodel(FLAGS.model_dir, input.get_input_reciever_fn())
            tf.logging.warn("save model to %s" % model_dir)

    #cpu solo
    else:
        print("执行*************************")
        estimator.train(train_input_fn, max_steps=FLAGS.max_steps, saving_listeners=listeners)
        dir = estimator.export_savedmodel(tf.flags.FLAGS.model_dir, input.get_input_reciever_fn())
        tf.logging.warn("save model to %s" % dir)

    for listener in listeners:
        print(listener.name)
        print(listener.history)

if __name__=="__main__":
    tf.app.run(main=bidaf_train,argv=None)
