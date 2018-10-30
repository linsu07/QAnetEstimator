import six
import tensorflow as tf
import time
from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverListener


'''
  * Created by linsu on 2017/12/10.
  * mailto: lsishere2002@hotmail.com
'''
class EvalListener(CheckpointSaverListener):
    def __init__(self, parent:tf.estimator.Estimator, input_fn,name = 'eval_data',hook=[]):

        self.parent = parent
        self.input_fn=input_fn
        self.name = name
        self.history = {}
        self.hook = hook
        # self.ema_enabled = ema_enabled
        # self.ema_decay = 0.9

    def after_save(self, session, global_step_value):
        accuracy_spec = self.parent.evaluate(input_fn=self.input_fn,name=self.name,hooks=self.hook)
 #       logging.info('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
#                                                               time.gmtime()))
 #       tf.logging.warn('evaluate %s at %s' % (self.name,time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime())))
        tf.logging.warn("evaluate "+ self.name + ":\n "+ ', '.join('%s = %s\n' % (k, v)
                        for k, v in sorted(six.iteritems(accuracy_spec))))

        step = accuracy_spec["global_step"]
        for k, v in sorted(six.iteritems(accuracy_spec)):
            if k == "global_step":
                continue
            self.history[k] = self.history.get(k, [])
            self.history[k].append((step, v))

class LoadEMAHook(tf.train.SessionRunHook):
    def __init__(self, model_dir,decay_rate):
        super(LoadEMAHook, self).__init__()
        self._model_dir = model_dir
        self.decay_rate = decay_rate

    def begin(self):
        ema = tf.train.ExponentialMovingAverage(self.decay_rate)
        variables_to_restore = ema.variables_to_restore()
        #print( "in hook-------------------------------------")
        #print(variables_to_restore)
        self._load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
            tf.train.latest_checkpoint(self._model_dir), variables_to_restore)

    def after_create_session(self, sess, coord):
        tf.logging.info('Reloading EMA...')
        self._load_ema(sess)