# https://www.reddit.com/r/tensorflow/comments/g5y1o5/implementation_of_evonorm_s0_and_b0_on_tensorflow/
import os
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0,1,2,3
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
# tf.config.run_functions_eagerly(True)
tf.random.set_seed(0)

physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)


DEFAULT_EPSILON_VALUE = 1e-5


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


class EvoNormS0(tf.keras.layers.Layer):
    def __init__(self, groups, axis=-1):
        # TODO make diff axis work
        super(EvoNormS0, self).__init__()
        self.groups, self.axis = groups, axis

    def build(self, input_shape):
        inlen = len(input_shape)
        shape = [1] * inlen
        shape[self.axis] = input_shape[self.axis]
        self.gamma = self.add_weight(name="gamma", shape=shape, initializer=tf.initializers.Ones())
        self.beta = self.add_weight(name="beta", shape=shape, initializer=tf.initializers.Zeros())
        self.v1 = self.add_weight(name="v1", shape=shape, initializer=tf.initializers.Ones())

        groups = min(input_shape[self.axis], self.groups)
        group_shape = input_shape.as_list()
        group_shape[self.axis] = input_shape[self.axis] // groups
        group_shape.insert(self.axis, groups)
        self.group_shape = tf.TensorShape(group_shape)

        std_shape = list(range(1, inlen+self.axis))
        std_shape.append(inlen)
        self.std_shape = tf.TensorShape(std_shape)

    @tf.function
    def call(self, inputs, training=True):
        grouped_inputs = tf.reshape(inputs, self.group_shape)
        _, var = tf.nn.moments(grouped_inputs, self.std_shape, keepdims=True)
        std = tf.sqrt(var + DEFAULT_EPSILON_VALUE)
        std = tf.broadcast_to(std, self.group_shape)
        group_std = tf.reshape(std, tf.shape(inputs))

        return (inputs * tf.sigmoid(self.v1 * inputs)) / group_std * self.gamma + self.beta

    # def get_config(self):
    #     config = {'group': self.groups,}
    #     base_config = super(EvoNormS0, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    # data_train, data_test = tf.keras.datasets.cifar10.load_data()

    # test = np.transpose(data_train[0])
    # test2 = np.transpose(data_train[1])
    # test3 = np.stack((test,test2), axis=0)
    # data_train = np.transpose(np.stack(data_train, axis=0))
    # data_test = np.transpose(np.stack(data_test, axis=0))
    # data_train = np.concatenate(data_train, data_test)
    # np.random.shuffle(data_train)

    train_images, train_labels, test_images, test_labels = np.float64(train_images), np.uint8(train_labels), np.float64(test_images), np.uint8(test_labels)
    # train_images, test_images = train_images / 255.0, test_images / 255.0

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(input_shape=(32, 32, 3), filters=16, kernel_size=(3, 3)),
    #     EvoNormS0(16, groups=8),
    #     tf.keras.layers.Flatten(),
    #     # tf.keras.layers.Dense(512, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(10)
    # ])
    # # model.summary()
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    # model.fit(train_images, train_labels, epochs=5)
    # model.evaluate(test_images, test_labels)



    class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            
            # self.layer_conv2d_in = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')
            # self.layer_conv2d_in = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), use_bias=False) # use_bias does not affect the output
            # self.layer_conv2d_in_evo = EvoNormS0(8)

            self.layer_flatten = tf.keras.layers.Flatten()

            # self.layer_dense_in = tf.keras.layers.Dense(512, activation='relu')
            # self.layer_dense_in = tf.keras.layers.Dense(512, use_bias=False)
            # self.layer_dense_in_evo = EvoNormS0(32)
            
            # self.layer_lstm_in = tf.keras.layers.LSTM(512)
            # self.layer_lstm_in = tf.keras.layers.LSTM(512, activation='linear', recurrent_activation='sigmoid', use_bias=False)
            # self.layer_lstm_in_evo = EvoNormS0(32)

            # self.layer_attn_in = tf.keras.layers.LSTM(512)
            # # self.layer_attn_in = tf.keras.layers.LSTM(512, activation='linear', recurrent_activation='sigmoid', use_bias=False)
            # # self.layer_attn_in_evo = EvoNormS0(32)

            # self.layer_dropout = tf.keras.layers.Dropout(0.2)
            self.layer_dense_logits_out = tf.keras.layers.Dense(10, use_bias=False)

        @tf.function
        def call(self, inputs, training=None):
            # out = self.layer_conv2d_in(inputs)
            # out = self.layer_conv2d_in_evo(out)
            # out = self.layer_flatten(out)
            out = self.layer_flatten(inputs)
            # out = self.layer_dense_in(out)
            # out = self.layer_dense_in_evo(out)
            # out = self.layer_lstm_in(tf.expand_dims(out, axis=1))
            # out = self.layer_lstm_in_evo(out)
            out = self.layer_attn_in(out)
            # out = self.layer_attn_in_evo(out)
            # out = self.layer_dropout(out, training=training)
            out = self.layer_dense_logits_out(out)
            return out

        _loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        _optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        @tf.function
        def train(self, inputs, targets):
            with tf.GradientTape() as tape:
                outputs = self(inputs, training=True)
                loss = self._loss(targets, outputs)
            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return outputs, loss
        @tf.function
        def test(self, inputs, targets):
            outputs = self(inputs, training=False)
            return outputs, self._loss(targets, outputs)
    model = MyModel()
    
    batch_size = 1000
    for epoc in range(5):
        # np.random.shuffle(train_images)

        data_size_train, loss_total_train = train_images.shape[0], 0.0
        for i in range(0, data_size_train, batch_size):
            outputs, loss = model.train(train_images[i:i+batch_size], train_labels[i:i+batch_size])
            loss_total_train += loss

        data_size_test, loss_total_test = test_images.shape[0], 0.0
        for i in range(0, data_size_test, batch_size):
            outputs, loss = model.test(test_images[i:i+batch_size], test_labels[i:i+batch_size])
            loss_total_test += loss
        
        print('#{} avg train loss {:.12f} avg test loss {:.12f}'.format(epoc, loss_total_train / (data_size_train / batch_size), loss_total_test / (data_size_test / batch_size)))

