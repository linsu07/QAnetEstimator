import tensorflow as tf
from tensorflow.contrib.layers import dense_to_sparse


class la(tf.layers.Layer):
    def __init__(self,name = "ab"):
        super(la, self).__init__( name= name)
    def build(self, _):
        print("in build")
        a = self.add_variable("a",[10,10])
        print(a)
        self.built = True
    def call(self, inputs, **kwargs):
        return inputs

if __name__=="__main__":
    tf.enable_eager_execution()
    input_ori = [[[[4,2,4]
         ,[2,10,9]
         ,[6,8,7]],
        [[1,2,4]
        ,[2,10,3]
        ,[6,8,7]]],
    [[[1,2,4]
         ,[2,3,9]
         ,[6,8,7]],
     [[1,2,4]
         ,[2,10,3]
         ,[6,20,7]]]]

    print(input_ori)

    input  = tf.matrix_band_part(input_ori, 0, -1)

    print(input)

    input = tf.matrix_band_part(input_ori, -1, 0)

    print(input)



    # context = ["我","爱","北 京","天 安 们"]
    #
    # context = tf.constant(context)
    #
    # ret = tf.string_split(context," ")
    #
    # tf.substr
    #
    # print(ret)

    # p2 = [[
    #     [1,1,1,1,0,0],[1,0,0,0,0,0]
    # ],[
    #     [1,1,1,0,0,0],[0,0,0,0,0,0]
    # ]]
    # p2t = tf.constant(p2)
    # p2t = tf.reshape(p2t,[2,12])
    #
    # len = tf.reduce_sum(tf.sign(tf.abs(p2t)),-1)
    # print(len)
    # max_len = tf.reduce_max(len)
    # print(max_len)
    # tmp = tf.constant(1,shape = [2,2])
    # print(tmp)
    # p_sparse = dense_to_sparse(p2t)
    # print(p_sparse)
    # p = tf.sparse_tensor_to_dense(p_sparse)
    # print(p)
    # #[2,2,3,3]
    # x = [[[[4,2,4]
    #      ,[2,10,9]
    #      ,[6,8,7]],
    #     [[1,2,4]
    #     ,[2,10,3]
    #     ,[6,8,7]]],
    # [[[1,2,4]
    #      ,[2,3,9]
    #      ,[6,8,7]],
    #  [[1,2,4]
    #      ,[2,10,3]
    #      ,[6,20,7]]]]
    # x_tensor = tf.constant(x,dtype=tf.int64)
    # [d1,d2,d3,d4]= tf.shape(x_tensor)
    # x_tensor_reshape = tf.reshape(x_tensor,[d1,d4*d2*d3])
    # max = tf.reduce_max(x_tensor_reshape,-1)
    # print(max)
    # indice = tf.expand_dims(tf.argmax(x_tensor_reshape,-1),-1)
    # print(indice)
    # row_index = tf.expand_dims(tf.constant(range(indice.shape[0]),dtype=tf.int64),-1)
    # print(row_index)
    # indice = tf.concat([row_index,indice],axis=-1)
    # print(indice)
    # only_max = tf.sparse_to_dense(indice,x_tensor_reshape.shape,max)
    # print(only_max)
    # only_max = tf.reshape(only_max,[d1,d2,d3,d4])
    # print(max)
    # loc = tf.where(tf.cast(only_max,tf.bool))
    # ret = tf.slice(loc,[0,1],[-1,-1])
    # print(ret)


    # ret = tf.concat([tf.cast(ret,tf.int64),org_max],axis=-1)
    # print (ret)
    # col_max = tf.argmax(x_tensor,-1)
    # row_max = tf.argmax(x_tensor,-2)
    #
    # print(col_max)
    # print(row_max)
    # l = la(None)
    # with tf.variable_scope("s1"):
    #     l(tf.constant(0))
    # with tf.variable_scope("s2"):
    #     l(tf.constant(1))