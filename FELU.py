import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense,Dropout, Softmax, ReLU

# Qiumei Z, Dan T, Fenghua W. Improved convolutional neural network based on fast exponentially linear unit activation
# function[J]. Ieee Access, 2019, 7: 151359-151367.
@tf.keras.utils.register_keras_serializable()
class FELU(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        super(FELU, self).__init__(name=name)
        super(FELU, self).__init__(**kwargs)

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.initializer = tf.constant_initializer(value=1.0)
        self.alpha = tf.Variable(initial_value=self.initializer(shape=[c], dtype=tf.float32),
                                       dtype=tf.float32,
                                       trainable=True)
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        pos = tf.nn.relu(inputs)
        neg = (inputs - tf.math.abs(inputs)) * 0.5
        neg_result = self.alpha * (2**(neg/math.log(2))-1)
        return pos + neg_result

    def get_config(self):
        base_config = super(FELU, self).get_config()
        return dict(list(base_config.items()))

