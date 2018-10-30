import tensorflow as tf
from tensorflow.contrib.layers import dense_to_sparse
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import prediction_keys, metric_keys
from tensorflow.python.estimator.canned.head import _Head, _classification_output, _DEFAULT_SERVING_KEY, \
    _PREDICT_SERVING_KEY, LossSpec, _summary_key
from tensorflow.python.estimator.export import export_output
from tensorflow.python.training import training_util
'''
  * Created by linsu on 2018/8/28.
  * mailto: lsishere2002@hotmail.com
'''

class SpanMatchHead(_Head):
    def __init__(self,name ="SpanMatchHead"):
        self._name = name
    @property
    def name(self):
        return self._name
    def create_estimator_spec(
            self, features, mode, logits, labels=None, optimizer=None,
            train_op_fn=None, regularization_losses=None,params= None):
        """Returns an `EstimatorSpec`.

        Args:
          features: Input `dict` of `Tensor` or `SparseTensor` objects.
          mode: Estimator's `ModeKeys`.
          logits: a tensor array with 2 element, one for start positon probilities
          , and one for the end postion probilities.
          the shape is `[batch_size, logits_dimension]`.
          labels: a tensor with demention [batch_size, 2], 2 position for true position in a doc
          optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
            Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
            updates variables and increments `global_step`.
          train_op_fn: Function that takes a scalar loss `Tensor` and returns
            `train_op`. Used if `optimizer` is `None`.
          regularization_losses: A list of additional scalar losses to be added to
            the training loss, such as regularization losses. These losses are
            usually expressed as a batch average, so for best results users need to
            set `loss_reduction=SUM_OVER_BATCH_SIZE` or
            `loss_reduction=SUM_OVER_NONZERO_WEIGHTS` when creating the head to
            avoid scaling errors.
        Returns:
          `EstimatorSpec`.
        Raises:
          ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
            mode, or if both are set.
        """
        with tf.name_scope(self._name, 'head'):
            # Predict.
            pred_keys = prediction_keys.PredictionKeys
            with tf.name_scope(None, 'predictions', (logits,)):

                [start_logits,end_logits]  = logits
                shape = tf.shape(start_logits)
                d1 = shape[0]
                d2 = shape[1]
                d3=  shape[2]
                start_logits = tf.reshape(tf.nn.softmax(tf.reshape(start_logits,[-1,d2*d3]),-1),[d1,d2,d3,1])
                end_logits =  tf.reshape(tf.nn.softmax(tf.reshape(end_logits,[-1,d2*d3]),-1),[d1,d2,1,d3])
                #[batch_size, sent_number, seq_len,seq_len]
                mul = tf.multiply(start_logits,end_logits)
                x_tensor_band = tf.matrix_band_part(mul, 0, params.ans_limit)
                x_tensor_reshape = tf.reshape(x_tensor_band,[d1,d2*d3*d3])
                #[batch_size]
                max = tf.reduce_max(x_tensor_reshape,-1)
                #[batch_size,1]
                row_index = tf.where(tf.not_equal(max,tf.ones_like(max)+2.1))
                indice = tf.expand_dims(tf.argmax(x_tensor_reshape,-1),-1)
                indice = tf.concat([row_index,indice],axis=-1)
                only_max = tf.sparse_to_dense(indice,tf.shape(x_tensor_reshape,out_type=tf.int64),max)
                only_max = tf.reshape(only_max,[d1,d2,d3,d3])
                loc = tf.where(tf.cast(only_max,tf.bool))
                loc = tf.slice(loc,[0,1],[-1,-1],name="locations") #[batch_size,postions=3] positions = (sent_no,start_pos,end_pos)
                predictions = {
                    "s_logits": start_logits,
                    "e_logits": end_logits,
                    pred_keys.PROBABILITIES: max,
                    "locations":loc
                }
            if mode == model_fn.ModeKeys.PREDICT:
                output = export_output.PredictOutput(predictions)
                return model_fn.EstimatorSpec(
                    mode=model_fn.ModeKeys.PREDICT,
                    predictions=predictions,
                    export_outputs={
                        _DEFAULT_SERVING_KEY: output,
                        _PREDICT_SERVING_KEY: output
                    })

            training_loss, unreduced_loss, weights, label_ids = self.create_loss(
                features=None, mode=mode, logits=logits, labels=labels)

            if regularization_losses:
                regularization_loss = tf.add_n(regularization_losses)
                regularized_training_loss = tf.add_n(
                    [training_loss, regularization_loss])
            else:
                regularization_loss = None
                regularized_training_loss = training_loss
            # Eval.
            if mode == model_fn.ModeKeys.EVAL:
                return model_fn.EstimatorSpec(
                    mode=model_fn.ModeKeys.EVAL,
                    predictions=predictions,
                    loss=regularized_training_loss,
                    eval_metric_ops=self._eval_metric_ops(
                        labels=label_ids,
                        predict=tf.sign(tf.abs(tf.reshape(only_max,[d1,d2*d3*d3]))),
                        location = loc,
                        pro = max,
                        unreduced_loss=unreduced_loss,
                        regularization_loss=regularization_loss))

            # Train.
            if optimizer is not None:
                if train_op_fn is not None:
                    raise ValueError('train_op_fn and optimizer cannot both be set.')
                train_op = optimizer.minimize(
                    regularized_training_loss,
                    global_step=training_util.get_global_step())
            elif train_op_fn is not None:
                train_op = train_op_fn(regularized_training_loss)
            else:
                raise ValueError('train_op_fn and optimizer cannot both be None.')
        with tf.name_scope(''):
            keys = metric_keys.MetricKeys
            tf.summary.scalar(
                _summary_key(self._name, keys.LOSS),
                regularized_training_loss)

            if regularization_loss is not None:
                tf.summary.scalar(
                    _summary_key(self._name, keys.LOSS_REGULARIZATION),
                    regularization_loss)
        return model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.TRAIN,
            predictions=predictions,
            loss=regularized_training_loss,
            train_op=train_op)

    def create_loss(self, features, mode, logits, labels):
        """See `Head`."""
        del mode,features  # Unused for this head.
        [start_logits,end_logits]  = logits
        [start_label,end_label] = labels
        if isinstance(start_label,tf.SparseTensor):
            start_label = tf.sparse_tensor_to_dense(start_label)
            end_label = tf.sparse_tensor_to_dense(end_label)

        shape= tf.shape(start_logits)
        d1 = shape[0]
        d2 = shape[1]
        d3 = shape[2]
        start_logits = tf.reshape(start_logits,[d1,d2*d3])
        end_logits = tf.reshape(end_logits,[d1,d2*d3])
        start_label = tf.reshape(start_label,[d1,d2*d3])
        end_label = tf.reshape(end_label,[d1,d2*d3])

        loss1 = tf.nn.softmax_cross_entropy_with_logits(labels=start_label,logits=start_logits)
        loss2 = tf.nn.softmax_cross_entropy_with_logits(labels=end_label,logits=end_logits)
        unreduced_loss = loss1+loss2
        loss = tf.reduce_mean(unreduced_loss)
        return LossSpec(
            training_loss=loss,
            unreduced_loss=unreduced_loss,
            weights=None,
            processed_labels=labels)

    def _eval_metric_ops(
            self, labels,predict, location,pro,unreduced_loss, regularization_loss):
        """Returns the Eval metric ops."""
        with tf.name_scope(
                None, 'metrics',
                (labels, unreduced_loss,predict, location, regularization_loss)):
            keys = metric_keys.MetricKeys
            [label_start,label_end] = labels
            shape = tf.shape(label_start)
            d1= shape[0]
            d2 = shape[1]
            d3 = shape[2]
            if isinstance(label_start,tf.SparseTensor):
                label_start = tf.sparse_tensor_to_dense(label_start)
                label_end = tf.sparse_tensor_to_dense(label_end)
            label_start = tf.expand_dims(label_start,-1)
            label_end = tf.expand_dims(label_end,-2)

            mul = tf.multiply(label_start,label_end)
            mul = tf.reshape(mul,[d1,d2*d3*d3])

            start_pos = tf.argmax(tf.reshape(label_start,[-1,d2*d3]),-1,output_type=tf.int32)
            end_pos = tf.argmax(tf.reshape(label_end,[-1,d2*d3]),-1,output_type=tf.int32)

            #[sentNo,start,end] = tf.squeeze(tf.split(tf.cast(location,tf.int32),3,axis=-1),-1)
            postions = tf.split(tf.cast(location,tf.int32),3,axis=-1)
            # squeezed = tf.shape(tf.squeeze(tf.split(tf.cast(location,tf.int32),3,axis=-1),-1))
            # print(squeezed)
            sentNo = tf.squeeze(postions[0],-1)
            start = tf.squeeze(postions[1],-1)
            end = tf.squeeze(postions[2],-1)
            start = start+sentNo*d2
            end = end+sentNo*d2

            min = tf.minimum(end_pos+1-start,end+1-start_pos)
            min = tf.minimum(min,end+1-start)
            min = tf.minimum(min,end_pos+1-start_pos)

            true_postive = tf.nn.relu(min)
            predict_scan = tf.nn.relu(end+1-start)
            predict_scan = tf.where(tf.cast(predict_scan,tf.bool),predict_scan,tf.ones_like(predict_scan)*(-1))
            label_scan = tf.nn.relu(end_pos+1-start_pos)
            label_scan = tf.where(tf.cast(label_scan,tf.bool),label_scan,tf.ones_like(label_scan)*(-1))
            labels = tf.argmax(mul,-1)
            predictions = tf.argmax(predict,-1)
            precision, precision_op = tf.metrics.mean(tf.nn.relu(true_postive/predict_scan))
            recall, recall_op = tf.metrics.mean(tf.nn.relu(true_postive/label_scan))

            def f1(precision,recall):
                return 2*precision*recall/(precision+recall)
            f1_value = f1(precision,recall)
            f1_op = f1(precision_op,recall_op)

            metric_ops = {
                # Estimator already adds a metric for loss.
                # TODO(xiejw): Any other metrics?
                _summary_key(self._name, keys.LOSS_MEAN):
                    tf.metrics.mean(
                        values=unreduced_loss,
                        weights=None,
                        name=keys.LOSS_MEAN),
                _summary_key(self._name, "EM"):
                    tf.metrics.accuracy(
                        labels=labels,
                        predictions=predictions,
                        weights=None,
                        name="EM"),
                _summary_key(self.name,keys.PRECISION):(precision, precision_op),
                _summary_key(self.name,keys.RECALL):(recall, recall_op),
                _summary_key(self.name,"f1_score"):(f1_value, f1_op),
                _summary_key(self.name,"raw_pro"): tf.metrics.mean(pro)
            }

            if regularization_loss is not None:
                metric_ops[_summary_key(self._name, keys.LOSS_REGULARIZATION)] = (
                    tf.metrics.mean(
                        values=regularization_loss,
                        name=keys.LOSS_REGULARIZATION))
        return metric_ops