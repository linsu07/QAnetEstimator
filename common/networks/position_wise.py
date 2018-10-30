import math

import tensorflow as tf
"""
PE(pos;2i) = sin(pos\10000**(2i\dmodel) )
PE(pos;2i+1) = cos(pos\10000**(2i\dmodel) )
Gets a bunch of sinusoids of different frequencies.
   Each channel of the input Tensor is incremented by a sinusoid of a different
   frequency and phase.
   This allows attention to learn to use absolute and relative positions.
   Timing signals should be added to some precursors of both the query and the
   memory inputs to attention.
   The use of relative position is possible because sin(x+y) and cos(x+y) can be
   experessed in terms of y, sin(x) and cos(x).
   In particular, we use a geometric sequence of timescales starting with
   min_timescale and ending with max_timescale.  The number of different
   timescales is equal to channels / 2. For each timescale, we
   generate the two sinusoidal signals sin(timestep/timescale) and
   cos(timestep/timescale).  All of these sinusoids are concatenated in
   the channels dimension.
   Args:
   length: scalar, length of timing signal sequence.
   channels: scalar, size of timing embeddings to create. The number of
       different timescales is equal to channels / 2.
   min_timescale: a float
   max_timescale: a float
   Returns:
   a Tensor of timing signals [1, length, channels]
   """

class PositionAwareLayer(tf.layers.Layer):
    def __init__(self, is_trainning=True
                 , name="PositionAwareLayer"
                 , dtype=tf.float32):
        super(PositionAwareLayer,self).__init__(is_trainning, name, dtype)

    def build(self, _):
        self.max_timescale = 1.0e4
        self.min_timescale = 1.0

    '''
    input must be [batch_size, seq_len, feature_size]
    '''
    # def call(self,input, **kwargs):
    #     mask = kwargs.get('mask')
    #     shape = tf.shape(input)
    #     seq_len,feature_size = shape[1],shape[2]
    #     time_scale_numbers = feature_size//2
    #     time_seq = 2*tf.range(time_scale_numbers)/feature_size
    #     tmp = tf.ones_like(time_seq)*self.max_timescale
    #     dim_scale = 1/tf.pow(tmp,time_seq)
    #     #[1,feature_size/2]
    #     dim_scale = tf.cast(tf.expand_dims(dim_scale,0),tf.float32)
    #     position = tf.to_float(tf.range(seq_len))
    #     scaled_time = tf.expand_dims(position, 1) * dim_scale
    #     sin = tf.expand_dims(tf.sin(scaled_time),-1)
    #     cos = tf.expand_dims(tf.cos(scaled_time),-1)
    #     signal = tf.concat([sin,cos],-1)
    #     # sin = tf.sin(scaled_time)
    #     # cos = tf.cos(scaled_time)
    #     # signal = tf.concat([sin,cos],axis=1)
    #     #signal = tf.pad(signal, [[0, 0], [0, tf.mod(feature_size, 2)]])
    #     signal = tf.reshape(signal, [1, seq_len, feature_size])
    #     ret = tf.add(signal,input)
    #     if mask!=None:
    #         ret = ret*mask
    #
    #     return ret

    def call(self, input, **kwargs):
        mask = kwargs.get('mask')
        shape = tf.shape(input)
        seq_len,feature_size = shape[1],shape[2]
        time_scale_numbers = feature_size//2
        log_timescale_increment = math.log(self.max_timescale/self.min_timescale)\
                                  /(tf.cast(time_scale_numbers,tf.float32)-1.0)
        inv_timescales = self.min_timescale * tf.exp(
            tf.to_float(tf.range(time_scale_numbers)) * -log_timescale_increment)

        position = tf.to_float(tf.range(seq_len))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)

        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

        signal = tf.pad(signal, [[0, 0], [0, tf.mod(feature_size, 2)]])
        signal = tf.reshape(signal, [1, seq_len, feature_size])
        feature = tf.add(signal,input)
        if mask!=None:
            return feature*mask
        else:
            return feature
