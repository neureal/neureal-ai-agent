import numpy as np
import tensorflow as tf



class DQNAgent:
    def __init__(self, actions, state_shape):
        self.actions = actions
        self.state_shape = state_shape
        self.create_model()

    def loss_func(self, actions):
        def nested_func(label, logits):
            print(label, logits)
            return tf.losses.mean_squared_error(tf.reduce_sum(tf.multiply(actions, logits), axis=-1), label)

        return nested_func

    def create_model(self):
        def loss_func(actions):
            def nested_func(label, logits):
                return tf.losses.mean_squared_error(tf.reduce_sum(tf.multiply(actions, logits), axis=-1),
                                                    tf.reduce_sum(label, axis=-1))

            return nested_func

        # Define both Input tensors.
        states = tf.keras.Input(self.state_shape)
        actions = tf.keras.Input((self.actions))
        # Define the sequential stack of layers to process states.
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.actions, activation=None),
        ])
        # Define a model that goes from (states, actions) to sequential(states).
        # Although at this stage actions is not used, it is thus feedable.
        self.model = tf.keras.Model([states, actions], model(states))
        # Compile the model (unchanged code) ; the fed actions will be used in practice.
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_func(actions))



if __name__ == '__main__':
    model = DQNAgent(4, (80, 80, 4))
    states = np.random.rand(100, 80, 80, 4)
    actions = np.random.randint(0, 4, (100, 4))
    label = np.random.randint(0, 4, (100, 4))
    # feed both states and actions to the model
    model.model.fit([states, actions], label)
