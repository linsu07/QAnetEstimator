import tensorflow as tf
class DimensionSizeControl(tf.layers.Layer):
    """
    目的: 控制输入tensor的某一维度大小
    示例: 希望将输入tensor x的axis=1限制大小为5
    x = tf.get_variable("x", shape=[2, 6, 10, 10], dtype=tf.float32, initializer=tf.random_normal_initializer())
    layer = DimensionSizeControl(axis=1, max_size=5)
    y = layer(x)
    y.shape == (2, 5, 10, 10)


    """
    def __init__(self, axis = 1, max_size = 20, control_tensor_names = None, trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):
        """
        :param axis: 指定输入tensor的第几个维度控制大小
        :param max_size: 指定控制大小阈值，大于改阈值时，对输入tensor做slice截断
        :param control_tensor_names: 输入tensor为dict嵌套时，通过这个参数指定dict中哪些key对应的tensor需要做维度控制
        :param trainable:
        :param name:
        :param dtype:
        :param activity_regularizer:
        :param kwargs:
        """
        super(DimensionSizeControl, self).__init__(trainable, name, dtype, activity_regularizer, **kwargs)
        self.control_tensor_names = control_tensor_names
        self.max_size = max_size
        self.struct_type = None
        self.control_shape_id = axis

    def get_slice_end(self, input_shape):
        end = []
        for index, item in enumerate(input_shape):
            if index == self.control_shape_id:
                end.append(self.max_size)
            elif item.value != None:
                end.append(item.value)
            else:
                end.append(-1)

        return tf.constant(end, dtype=tf.int32)

    def build(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            assert len(input_shape) > self.control_shape_id, "参数 axis 应该小于inputs总shape数"
            self.struct_type = "tensor"
            self.slice_start = tf.constant([0] * len(input_shape), dtype=tf.int32)
            self.slice_end = self.get_slice_end(input_shape)

        elif isinstance(input_shape, list):
            self.struct_type = "tensor_list"
            raise ValueError("DimensionSizeControl 暂不支持list(tensor)，敬请期待")

        elif isinstance(input_shape, dict):
            assert self.control_tensor_names != None, "DimensionSizeControl的输入tensor是dict类型时，需指明哪些key对应的tensor需要被控制大小"
            self.struct_type = "tensor_dict"
            self.control_tensor_names = set(self.control_tensor_names)
            self.start_dict = {}
            self.end_dict = {}
            for tensor_name, tensor_shape in input_shape.items():
                if tensor_name in self.control_tensor_names:
                    self.start_dict[tensor_name] = tf.constant([0] * len(tensor_shape), dtype=tf.int32)
                    self.end_dict[tensor_name] = self.get_slice_end(tensor_shape)

        else:
            raise ValueError("DimensionSizeControl 仅支持tensor/list(tensor)/dict(tensor)，不支持的输入类型: " + str(type(input_shape)))
        self.built = True

    def call(self, inputs, **kwargs):
        if self.struct_type == "tensor":
            origin_shape = tf.shape(inputs)
            return tf.cond(
                origin_shape[self.control_shape_id] > self.max_size,
                lambda : tf.slice(inputs, self.slice_start, self.slice_end),
                lambda : inputs,
                name="DimensionSizeControlOutput"
                )

        if self.struct_type == "tensor_dict":
            origin_shapes_dict = {key: tf.shape(value) for key, value in inputs.items() if key in self.control_tensor_names}
            for tensor_name, tensor_shape in origin_shapes_dict.items():
                inputs[tensor_name] = tf.cond(
                    tensor_shape[self.control_shape_id] > self.max_size,
                    lambda : tf.slice(inputs[tensor_name], self.start_dict[tensor_name], self.end_dict[tensor_name]),
                    lambda : inputs[tensor_name],
                    name="DimensionSizeControlOutput"
                )
        return inputs

    def __call__(self, inputs, **kwargs):
        return super(DimensionSizeControl, self).__call__(inputs=inputs, **kwargs)

def tensor_test():
    with tf.variable_scope("tensor_test"):
        x = tf.get_variable("x", shape=[2, 6, 10, 10], dtype=tf.float32, initializer=tf.random_normal_initializer())
    init = tf.global_variables_initializer()
    layer = DimensionSizeControl(axis=1, max_size=5)
    y = layer(x)

    with tf.Session() as sess:
        sess.run(init)
        x_ = sess.run(x)
        tf.logging.debug("before SizeControl layer")
        tf.logging.debug(x_.shape)
        y_ = sess.run(y)
        tf.logging.debug("after SizeControl layer")
        tf.logging.debug(y_.shape)
    tf.logging.info("tensor_test is passed")

def tensor_list_test():
    with tf.variable_scope("tensor_list_test"):
        x1 = tf.get_variable("x1", shape=[2, 6, 10], dtype=tf.float32, initializer=tf.random_normal_initializer())
        x2 = tf.get_variable("x2", shape=[2, 6, 20], dtype=tf.float32, initializer=tf.random_normal_initializer())

    init = tf.global_variables_initializer()
    layer = DimensionSizeControl(axis=1, max_size=5)
    y = layer([x1, x2])

    with tf.Session() as sess:
        sess.run(init)
        x1_, x2_ = sess.run([x1, x2])
        tf.logging.debug(x1_.shape, x2_.shape)
        y_ = sess.run(y)
        tf.logging.debug(y_.shape)
    tf.logging.info("tensor_list_test is passed")

def tensor_dict_test():
    with tf.variable_scope("tensor_dict_test"):
        x1 = tf.get_variable("x1", shape=[2, 6, 10, 10], dtype=tf.float32, initializer=tf.random_normal_initializer())
        x2 = tf.get_variable("x2", shape=[2, 6, 20], dtype=tf.float32, initializer=tf.random_normal_initializer())
        x3 = tf.get_variable("x3", shape=[2, 1], dtype=tf.float32, initializer=tf.random_normal_initializer())

    init = tf.global_variables_initializer()
    layer = DimensionSizeControl(axis=1, max_size=4, control_tensor_names=["x1", "x2"])
    inputs = {"x1": x1, "x2": x2, "x3": x3}


    with tf.Session() as sess:
        sess.run(init)
        x_ = sess.run(inputs)
        tf.logging.debug("before DimensionSizeControl layer")
        for k, v in x_.items():
            tf.logging.debug(k)
            tf.logging.debug(v.shape)
        y = layer(inputs)
        y_ = sess.run(y)
        tf.logging.debug("after DimensionSizeControl layer")
        for k, v in y_.items():
            tf.logging.debug(k)
            tf.logging.debug(v.shape)
    tf.logging.info("tensor_dict_test is passed")



if __name__ == "__main__":
    tf.logging.set_verbosity("DEBUG")
    # tf.logging.set_verbosity("INFO")

    tensor_test()
    tensor_dict_test()

