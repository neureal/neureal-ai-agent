# https://www.reddit.com/r/tensorflow/comments/g5y1o5/implementation_of_evonorm_s0_and_b0_on_tensorflow/
import tensorflow as tf


DEFAULT_EPSILON_VALUE = 1e-5


def instance_std(x, eps=DEFAULT_EPSILON_VALUE):
    _, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    return tf.sqrt(var + eps)


def group_std(inputs, groups=32, eps=DEFAULT_EPSILON_VALUE, axis=-1):
    groups = min(inputs.shape[axis], groups)

    input_shape = tf.shape(inputs)
    group_shape = [input_shape[i] for i in range(4)]
    group_shape[axis] = input_shape[axis] // groups
    group_shape.insert(axis, groups)
    group_shape = tf.stack(group_shape)
    grouped_inputs = tf.reshape(inputs, group_shape)
    _, var = tf.nn.moments(grouped_inputs, [1, 2, 4], keepdims=True)

    std = tf.sqrt(var + eps)
    std = tf.broadcast_to(std, tf.shape(grouped_inputs))
    return tf.reshape(std, input_shape)


class EvoNormB0(tf.keras.layers.Layer):
    def __init__(self, channels, momentum=0.9, epsilon=DEFAULT_EPSILON_VALUE):
        super(EvoNormB0, self).__init__()

        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma = self.add_weight(name="gamma", shape=(1, 1, 1, channels), initializer=tf.initializers.Ones())
        self.beta = self.add_weight(name="beta", shape=(1, 1, 1, channels), initializer=tf.initializers.Zeros())
        self.v_1 = self.add_weight(name="v1", shape=(1, 1, 1, channels), initializer=tf.initializers.Ones())

        self.running_average_std = self.add_variable(trainable=False, shape=(1, 1, 1, channels), initializer=tf.initializers.Ones())

    def call(self, inputs, training=True):
        var = self.running_average_std
        if training:
            _, var = tf.nn.moments(inputs, [0, 1, 2], keepdims=True)
            self.running_average_std.assign(self.momentum * self.running_average_std + (1 - self.momentum) * var)
        else:
            pass

        denominator = tf.maximum(
            instance_std(inputs) + self.v_1 * inputs,
            tf.sqrt(var + self.epsilon),
            )
        return inputs * self.gamma / denominator + self.beta


class EvoNormS0(tf.keras.layers.Layer):
    def __init__(self, channels, groups=8):
        super(EvoNormS0, self).__init__()

        self.groups = groups

        self.gamma = self.add_weight(name="gamma", shape=(1, 1, 1, channels), initializer=tf.initializers.Ones())
        self.beta = self.add_weight(name="beta", shape=(1, 1, 1, channels), initializer=tf.initializers.Zeros())
        self.v_1 = self.add_weight(name="v1", shape=(1, 1, 1, channels), initializer=tf.initializers.Ones())

    def call(self, inputs, training=True):
        return (inputs * tf.sigmoid(self.v_1 * inputs)) / group_std(inputs, groups=self.groups) * self.gamma + self.beta

    def get_config(self):
        config = {'group': self.groups,}
        base_config = super(EvoNormS0, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(32, 32, 3), filters=16, kernel_size=(3, 3)),
        EvoNormS0(channels=16, groups=8)
    ])
    model.summary()
