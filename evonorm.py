import time, os
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0,1,2,3
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.config.run_functions_eagerly(True)
tf.random.set_seed(0)
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)

# TODO put these in a "tensorflow_neureal" or "wilutil" library, import tensorflow_neureal as tfn, tfn.fixinfnan(), tfn.EvoNormS0()

# @tf.function
# def fixinfnan(inputs):
#     zero = tf.constant(0.0, dtype=inputs.dtype)
#     isinf = tf.math.is_inf(inputs)
#     isneg = tf.math.equal(tf.math.sign(inputs),-1.0)
#     ispos = tf.math.logical_not(isneg)
#     isninf, ispinf = tf.math.logical_and(isinf, isneg), tf.math.logical_and(isinf, ispos)
#     inputs = tf.where(ispinf, zero, inputs) # inf = 0.0
#     inputs = tf.where(tf.math.logical_or(tf.math.is_nan(inputs), isninf), inputs.dtype.min, inputs) # nan = tf.float32.min, -inf = tf.float32.min
#     return inputs
@tf.function
def fixinfnan(inputs, replace):
    isinfnan = tf.math.logical_or(tf.math.is_nan(inputs), tf.math.is_inf(inputs))
    return tf.where(isinfnan, replace, inputs)


# https://www.reddit.com/r/tensorflow/comments/g5y1o5/implementation_of_evonorm_s0_and_b0_on_tensorflow/

# DEFAULT_EPSILON_VALUE = 1e-5
# def instance_std(x, eps=DEFAULT_EPSILON_VALUE):
#     _, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
#     return tf.sqrt(var + eps)


# def group_std(inputs, group_shape, std_shape, eps=DEFAULT_EPSILON_VALUE):
#     grouped_inputs = tf.reshape(inputs, group_shape)
#     _, var = tf.nn.moments(grouped_inputs, std_shape, keepdims=True)
#     std = tf.sqrt(var + eps)
#     std = tf.broadcast_to(std, group_shape)
#     return tf.reshape(std, tf.shape(inputs))


# class EvoNormB0(tf.keras.layers.Layer):
#     def __init__(self, channels, momentum=0.9, epsilon=DEFAULT_EPSILON_VALUE):
#         super(EvoNormB0, self).__init__()

#         self.momentum = momentum
#         self.epsilon = epsilon

#         self.gamma = self.add_weight(name="gamma", shape=(1, 1, 1, channels), initializer=tf.initializers.Ones())
#         self.beta = self.add_weight(name="beta", shape=(1, 1, 1, channels), initializer=tf.initializers.Zeros())
#         self.v1 = self.add_weight(name="v1", shape=(1, 1, 1, channels), initializer=tf.initializers.Ones())

#         self.running_average_std = self.add_weight(trainable=False, shape=(1, 1, 1, channels), initializer=tf.initializers.Ones())

#     def call(self, inputs, training=True):
#         var = self.running_average_std
#         if training:
#             _, var = tf.nn.moments(inputs, [0, 1, 2], keepdims=True)
#             self.running_average_std.assign(self.momentum * self.running_average_std + (1 - self.momentum) * var)
#         else:
#             pass

#         denominator = tf.maximum(
#             instance_std(inputs) + self.v1 * inputs,
#             tf.sqrt(var + self.epsilon),
#             )
#         return inputs * self.gamma / denominator + self.beta

#     def get_config(self):
#         config = {'group': self.groups,}
#         base_config = super(EvoNormB0, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


# TODO double check side effects and change list to TensorArray (list.append())
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
        self.group_shape[0].assign(input_shape[0])
        grouped_inputs = tf.reshape(inputs, self.group_shape)
        _, var = tf.nn.moments(grouped_inputs, self.std_shape, keepdims=True)
        std = tf.sqrt(var + self.eps)
        std = tf.broadcast_to(std, self.group_shape)
        group_std = tf.reshape(std, input_shape)

        return (inputs * tf.sigmoid(self.v1 * inputs)) / group_std * self.gamma + self.beta

    # def get_config(self):
    #     config = {'group': self.groups,}
    #     base_config = super(EvoNormS0, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, train_labels, test_images, test_labels = np.expand_dims(train_images,-1), np.expand_dims(train_labels,-1), np.expand_dims(test_images,-1), np.expand_dims(test_labels,-1) # use with mnist
    train_images, train_labels, test_images, test_labels = np.float64(train_images), np.int32(train_labels), np.float64(test_images), np.int32(test_labels)
    # train_images, test_images = train_images / 255.0, test_images / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3)),
        EvoNormS0(groups=8),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    # model.summary()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=3)
    model.evaluate(test_images, test_labels)
