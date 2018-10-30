import tensorflow as tf
import os

'''
  * Created by linsu on 2018/4/18.
  * mailto: lsishere2002@hotmail.com
'''

def get_vocab_file_size(file_name:str):
    lines = tf.gfile.Open(file_name).readlines()
    voc_size = len(lines)
    return  voc_size