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
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)

@tf.function
def fixinfnan(t):
    zero = tf.constant(0.0, dtype=tf.float64)
    isinf = tf.math.is_inf(t)
    isneg = tf.math.equal(tf.math.sign(t),-1.0)
    ispos = tf.math.logical_not(isneg)
    isninf, ispinf = tf.math.logical_and(isinf, isneg), tf.math.logical_and(isinf, ispos)
    t = tf.where(ispinf, zero, t) # inf = 0.0
    t = tf.where(tf.math.logical_or(tf.math.is_nan(t), isninf), tf.float64.min, t) # nan = tf.float32.min, -inf = tf.float32.min
    return t

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


class EvoNormS0(tf.keras.layers.Layer):
    def __init__(self, groups, eps=1e-5, axis=-1, name=None):
        # TODO make diff axis work
        super(EvoNormS0, self).__init__(name=name)
        self.groups, self.eps, self.axis = groups, eps, axis

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
        std = tf.sqrt(var + self.eps)
        std = tf.broadcast_to(std, self.group_shape)
        group_std = tf.reshape(std, tf.shape(inputs))

        return (inputs * tf.sigmoid(self.v1 * inputs)) / group_std * self.gamma + self.beta

    # def get_config(self):
    #     config = {'group': self.groups,}
    #     base_config = super(EvoNormS0, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    # (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # train_images, train_labels, test_images, test_labels = np.expand_dims(train_images,-1), np.expand_dims(train_labels,-1), np.expand_dims(test_images,-1), np.expand_dims(test_labels,-1)
    train_images, train_labels, test_images, test_labels = np.float64(train_images), np.int32(train_labels), np.float64(test_images), np.int32(test_labels)
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
            # use_bias does not affect the output of EvoNormS0
            self.categories = 10

            num_components, event_shape = 12, 1
            params_size = self.categories # Categorical
            params_size = tfp.layers.MixtureSameFamily.params_size(num_components,
                component_params_size=tfp.layers.MultivariateNormalTriL.params_size(event_shape))
            
            # self.layer_conv2d_in = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')
            # # self.layer_conv2d_in = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu')
            self.layer_conv2d_in = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), use_bias=False)
            self.layer_conv2d_in_evo = EvoNormS0(16)
            # # self.layer_conv2d_01 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu')
            self.layer_conv2d_01 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), use_bias=False)
            self.layer_conv2d_01_evo = EvoNormS0(16)
            # self.layer_conv2d_02 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), activation='relu')
            self.layer_conv2d_02 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), use_bias=False)
            self.layer_conv2d_02_evo = EvoNormS0(16)
            # self.layer_conv2d_03 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu')
            self.layer_conv2d_03 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), use_bias=False)
            self.layer_conv2d_03_evo = EvoNormS0(16)
            # # self.layer_conv2d_04 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu')
            self.layer_conv2d_04 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), use_bias=False)
            self.layer_conv2d_04_evo = EvoNormS0(16)
            # self.layer_conv2d_05 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(2, 2), activation='relu')
            self.layer_conv2d_05 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(2, 2), use_bias=False)
            self.layer_conv2d_05_evo = EvoNormS0(16)

            # self.layer_conv2d_10 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu')
            self.layer_conv2d_10 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), use_bias=False)
            self.layer_conv2d_10_evo = EvoNormS0(16)
            # self.layer_conv2d_11 = tf.keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), activation='relu')
            self.layer_conv2d_11 = tf.keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), use_bias=False)
            self.layer_conv2d_11_evo = EvoNormS0(16)
            self.layer_conv2d_12 = tf.keras.layers.Conv2D(filters=params_size, kernel_size=(1, 1), strides=(1, 1), use_bias=False)
            # self.layer_conv2d_12_evo = EvoNormS0(int(params_size/5))


            # self.layer_flatten = tf.keras.layers.Flatten()
            
            # self.layer_dense_01 = tf.keras.layers.Dense(1024, use_bias=False)
            # self.layer_dense_01_evo = EvoNormS0(32)

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
            self.layer_globalavg_logits_out = tf.keras.layers.GlobalAveragePooling2D()
            # self.layer_dense_logits_out = tf.keras.layers.Dense(params_size, use_bias=False)
            
            self.dist = tfp.layers.MixtureSameFamily(num_components, tfp.layers.MultivariateNormalTriL(event_shape))

        @tf.function
        def call(self, inputs, training=None):
            out = self.layer_conv2d_in(inputs)
            out = self.layer_conv2d_in_evo(out)
            out = self.layer_conv2d_01(out)
            out = self.layer_conv2d_01_evo(out)
            out = self.layer_conv2d_02(out)
            out = self.layer_conv2d_02_evo(out)
            out = self.layer_conv2d_03(out)
            out = self.layer_conv2d_03_evo(out)
            out = self.layer_conv2d_04(out)
            out = self.layer_conv2d_04_evo(out)
            out = self.layer_conv2d_05(out)
            out = self.layer_conv2d_05_evo(out)
            out = self.layer_conv2d_10(out)
            out = self.layer_conv2d_10_evo(out)
            out = self.layer_conv2d_11(out)
            out = self.layer_conv2d_11_evo(out)
            out = self.layer_conv2d_12(out)
            # out = self.layer_conv2d_12_evo(out)
            # out = self.layer_flatten(out)
            # out = self.layer_dense_01(out)
            # out = self.layer_dense_01_evo(out)
            # out = self.layer_flatten(inputs)
            # out = self.layer_dense_in(out)
            # out = self.layer_dense_in_evo(out)
            # out = self.layer_lstm_in(tf.expand_dims(out, axis=1))
            # out = self.layer_lstm_in_evo(out)
            # out = self.layer_attn_in(out)
            # out = self.layer_attn_in_evo(out)
            # out = self.layer_dropout(out, training=training)
            out = self.layer_globalavg_logits_out(out)
            # out = self.layer_dense_logits_out(out)
            return out

        # _loss_scc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        def _loss(self, targets, outputs):
            # dist = tfp.distributions.Categorical(logits=outputs)
            # loss = dist.log_prob(tf.squeeze(targets, axis=-1)) # Categorical
            dist = self.dist(outputs)
            loss = dist.log_prob(tf.cast(targets, dtype=tf.float64))
            loss = fixinfnan(loss)

            # loss = -tf.math.reduce_mean(loss)
            # loss = tf.math.abs(loss)
            loss = tf.math.negative(loss)

            onehot = tf.one_hot(tf.squeeze(targets, axis=-1), self.categories)
            softmax = tf.transpose(tf.map_fn(fn=lambda v:dist.prob(v), elems=tf.range(self.categories, dtype=tf.float64)))

            return loss, onehot, tf.clip_by_value(softmax, 0.0, 1.0)

        _optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        # _optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)
        metric_train_loss = tf.keras.metrics.Mean()
        @tf.function
        def train(self, inputs, targets):
            with tf.GradientTape() as tape:
                outputs = self(inputs, training=True)
                loss = self._loss(targets, outputs)[0]
            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            self.metric_train_loss.update_state(tf.math.reduce_mean(loss))
            return outputs
        
        metric_test_loss, metric_test_acc, metric_test_auc = tf.keras.metrics.Mean(), tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()
        @tf.function
        def test(self, inputs, targets):
            outputs = self(inputs, training=False)
            loss, onehot, softmax = self._loss(targets, outputs)

            self.metric_test_loss.update_state(tf.math.reduce_mean(loss))
            self.metric_test_acc.update_state(onehot, softmax)
            self.metric_test_auc.update_state(onehot, softmax)
            return outputs

    model = MyModel()
    
    epocs = 10
    batch_size = 10
    metric_train_loss, metric_test_loss, metric_test_acc, metric_test_auc = [], [], [], []
    for epoc in range(epocs):

        model.metric_train_loss.reset_states()
        for i in range(0, train_images.shape[0], batch_size):
            outputs = model.train(train_images[i:i+batch_size], train_labels[i:i+batch_size])
            metric_train_loss.append(model.metric_train_loss.result())

        model.metric_test_loss.reset_states(); model.metric_test_acc.reset_states(); model.metric_test_auc.reset_states()
        for i in range(0, test_images.shape[0], batch_size):
            outputs = model.test(test_images[i:i+batch_size], test_labels[i:i+batch_size])
            metric_test_loss.append(model.metric_test_loss.result()); metric_test_acc.append(model.metric_test_acc.result()); metric_test_auc.append(model.metric_test_auc.result())
        
        print('#{} train loss {:.12f} test loss {:.12f} test acc {:.12f} test auc {:.12f}'.format(epoc, model.metric_train_loss.result(), model.metric_test_loss.result(), model.metric_test_acc.result(), model.metric_test_auc.result()))

    plt.figure(num='evonorm', figsize=(24, 16), tight_layout=True)
    x = np.arange(len(metric_test_loss))
    ax3 = plt.subplot2grid((3, 1), (2, 0))
    plt.plot(x, metric_test_loss, label='metric_test_loss')
    plt.ylabel('value'); plt.xlabel('train step'); plt.legend(loc='upper left')
    ax2 = plt.subplot2grid((3, 1), (1, 0), sharex=ax3)
    plt.plot(x, metric_test_acc, label='metric_test_acc')
    plt.ylabel('value'); plt.xlabel('train step'); plt.legend(loc='upper left')
    ax1 = plt.subplot2grid((3, 1), (0, 0), sharex=ax3)
    plt.plot(x, metric_test_auc, label='metric_test_auc')
    plt.ylabel('value'); plt.xlabel('train step'); plt.legend(loc='upper left')
    plt.title('evonorm'); plt.show()