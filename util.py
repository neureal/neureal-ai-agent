import numpy as np
import tensorflow as tf
import gym

# TODO put this in "wilutil" library

def gym_flatten(self, space, x, dtype=np.float32):
    if isinstance(space, gym.spaces.Tuple): return np.concatenate([self._flatten(s, x_part, dtype) for x_part, s in zip(x, space.spaces)])
    elif isinstance(space, gym.spaces.Dict): return np.concatenate([self._flatten(s, x[key], dtype) for key, s in space.spaces.items()])
    elif isinstance(space, gym.spaces.Discrete):
        onehot = np.zeros(space.n, dtype=dtype); onehot[x] = 1.0; return onehot
    else: return np.asarray(x, dtype=dtype).flatten()

def print_time(t):
    days=int(t//86400);hours=int((t-days*86400)//3600);mins=int((t-days*86400-hours*3600)//60);secs=int((t-days*86400-hours*3600-mins*60))
    return "{:4d}:{:02d}:{:02d}:{:02d}".format(days,hours,mins,secs)

def replace_infnan(inputs, replace):
    isinfnan = tf.math.logical_or(tf.math.is_nan(inputs), tf.math.is_inf(inputs))
    return tf.where(isinfnan, replace, inputs)

class EvoNormS0(tf.keras.layers.Layer):
    def __init__(self, groups, eps=None, axis=-1, name=None):
        super(EvoNormS0, self).__init__(name=name)
        self.groups, self.axis = groups, axis
        if eps is None: eps = tf.experimental.numpy.finfo(self.compute_dtype).eps
        self.eps = tf.constant(eps, dtype=self.compute_dtype)

    def build(self, input_shape):
        inlen = len(input_shape)
        shape = [1] * inlen
        shape[self.axis] = input_shape[self.axis]
        self.gamma = self.add_weight(name="gamma", shape=shape, initializer=tf.keras.initializers.Ones())
        self.beta = self.add_weight(name="beta", shape=shape, initializer=tf.keras.initializers.Zeros())
        self.v1 = self.add_weight(name="v1", shape=shape, initializer=tf.keras.initializers.Ones())

        groups = min(input_shape[self.axis], self.groups)
        self.group_shape = input_shape.as_list()
        self.group_shape[self.axis] = input_shape[self.axis] // groups
        self.group_shape.insert(self.axis, groups)
        self.group_shape = tf.Variable(self.group_shape, trainable=False)

        std_shape = list(range(1, inlen+self.axis))
        std_shape.append(inlen)
        self.std_shape = tf.constant(std_shape)

    @tf.function
    def call(self, inputs, training=True):
        input_shape = tf.shape(inputs)
        self.group_shape[0].assign(input_shape[0]) # use same learned parameters with different batch size
        grouped_inputs = tf.reshape(inputs, self.group_shape)
        _, var = tf.nn.moments(grouped_inputs, self.std_shape, keepdims=True)
        std = tf.sqrt(var + self.eps)
        std = tf.broadcast_to(std, self.group_shape)
        group_std = tf.reshape(std, input_shape)

        return (inputs * tf.sigmoid(self.v1 * inputs)) / group_std * self.gamma + self.beta
