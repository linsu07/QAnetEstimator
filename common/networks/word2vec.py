from tensorflow.python.ops.init_ops import Initializer
import tensorflow as tf
import numpy as np

'''
  * Created by linsu on 2018/5/24.
  * mailto: lsishere2002@hotmail.com
'''
class word_embedding_initializer(Initializer):

    currentpath = None
    currentdata = None

    def __init__(self,np_file_path:str,dtype=tf.float32,include_word = True, vector_length=300):
        self.path = np_file_path
        self.dtype = dtype
        self.fb_len = 0
        self.local_file = "/tmp/dinfo_npdata"
        self.include_word = include_word
        self.vector_length = vector_length
        self.padding = np.zeros([1,vector_length],np.float32)

    def get_config(self):
        return {"path": self.path, "dtype": self.dtype.name,"include_word":self.include_word}
    def __call__(self, shape, dtype=None, partition_info=None):
        tf.logging.info("try init embedding from " + self.path)

        if self.path != word_embedding_initializer.currentpath:

            ## test
            if self.include_word:
                npdata = np.loadtxt(tf.gfile.GFile(self.path, "r"), usecols=(i  for i in range(1,self.vector_length + 1)),encoding = "utf-8")
            else:
                npdata = np.loadtxt(tf.gfile.GFile(self.path, "r"), usecols=(i  for i in range(0,self.vector_length)))

            word_embedding_initializer.currentdata = np.concatenate((self.padding,npdata),axis=0)
            word_embedding_initializer.currentpath = self.path
            return word_embedding_initializer.currentdata
        else:
            return word_embedding_initializer.currentdata