import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense,Dropout, Softmax, ReLU

@tf.keras.utils.register_keras_serializable()
class PReLU(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        super(PReLU, self).__init__(name=name)
        super(PReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        b, w, h, c = input_shape
        self.initializer = tf.constant_initializer(value=0.25)
        self.coefficient = tf.Variable(initial_value=self.initializer(shape=[c], dtype=tf.float32),
                                       dtype=tf.float32,
                                       trainable=True)
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        pos = tf.nn.relu(inputs)
        neg = self.coefficient * (inputs - tf.math.abs(inputs)) * 0.5
        return pos + neg

    def get_config(self):
        base_config = super(PReLU, self).get_config()
        return dict(list(base_config.items()))