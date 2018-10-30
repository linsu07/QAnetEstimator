import  tensorflow as tf
import collections
from comprehension.network.model import model_fn
from comprehension.parameter import user_params
tf.enable_eager_execution()
params = user_params(
         # procedure=[
         #     "wordembedding","contextual"
         #     ,"bidaf","modeling","output"] #output_att
        procedure=[
            "wordembedding","QAnetEmbedding"
            ,"dcn","QaModelBlock","QAOutput"]
        ,label_name="label"
        ,learning_rate=0.001
        ,embed_size=5
        ,embedding_file_path=None,
        context_name="context"
        ,question_name="question",
        rnn_hidden_size =3,
        data_dir="/tmp/bidaf/data"
        ,model_dir="/tmp/bidaf/model"
        ,batch_size=2,
        drop_out_rate=0.5
        ,p1 = "p1"
        ,p2="p2"
        ,feature_voc_file_path = None
        ,gpu_cores_list= None
        ,transfromer_conv_layers = 4
        ,transfromer_conv_kernel_size = 5
        ,transfromer_head_number = 2
        ,tansformer_d_model = 8
        ,clip_norm = 5.0
        ,use_char_embedding = 0
        ,char_embedding_size = 5
        ,char_feature_name = "c_f"
        ,char_question_name = "c_q"
        ,enable_ema = 0
        ,example_max_length = 100
        ,char_filters =  1
        ,ema_decay = 0.99
        ,ans_limit = 2

)
params.embedding_size = 5
params.feature_voc_file_len = 5
params.feature_voc_file = "/tmp/test/voc_file.txt"
params.embedding_file = None


context = [[["我","爱","北京","天安门","<padding>","<padding>"],["<padding>","<padding>","<padding>","<padding>","<padding>","<padding>"]]
    ,[["蓝天","爱","白云","<padding>","<padding>","<padding>"],["大地","河流","采矿","搬运","玩","<padding>"]]]

char_context = [[["a,',b","c","c","a,b,f,h,k,l,m,n","0","0"],["a,b,f","a,b,f","a,b,f","a,b,f","a,b,f","a,b,f"]]
    ,[["a,b,f","a,b,f","a,b,f","0","0","0"],["a,b,f","a,b,f","a,b,f","a,b,f","a,b,f","0"]]]

p1 = [[
    [0,0,1,0,0,0],[0,0,0,0,0,0]
],[
    [0,1,0,0,0,0],[0,0,0,0,0,0]
]]
p2 = [[
    [0,0,0,1,0,0],[0,0,0,0,0,0]
],[
    [0,0,1,0,0,0],[0,0,0,0,0,0]
]]
question = [["蓝天","爱","什么","<padding>"],
            ["白云","是","什么","<padding>"]
            ]
char_question = [["s,u,g","a,b,f,h,k,l,m,n","s,u,g","0"],
                 ["s,u,g","a,b,f,h,k,l,m,n","s,u,g","0"]
                 ]
features = {
    params.context_name:tf.constant(context),
    params.question_name:tf.constant(question),
    params.char_question_name:tf.constant(char_question),
    params.char_feature_name:tf.constant(char_context),
    "context_len":20
}
labels = {
    params.p1:tf.constant(p1),
    params.p2:tf.constant(p2)
}
config = collections.namedtuple("config",field_names=[])
config.num_ps_replicas = None
if __name__ == "__main__":
    spec = model_fn(features,labels,tf.estimator.ModeKeys.TRAIN,config,params)
    # with tf.Session() as sess:
    #     init = tf.global_variables_initializer()
    #     init2 = tf.tables_initializer()
    #     sess.run(init)
    #     sess.run(init2)
    #     print(sess.run([spec.loss,spec.train_op]))
