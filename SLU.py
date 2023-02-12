import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense,Dropout, Softmax, ReLU

# A novel softplus linear unit for deep convolutional neural networks
@tf.keras.utils.register_keras_serializable()
class SLU(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        super(SLU, self).__init__(name=name)
        super(SLU, self).__init__(**kwargs)

    def build(self, input_shape):
        b, w, h, c = input_shape
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        pos = tf.nn.relu(inputs)
        neg = (inputs - tf.math.abs(inputs)) * 0.5
        neg_computing = 2 * tf.math.log((tf.math.exp(neg)+1)/2)
        return pos + neg_computing

    def get_config(self):
        base_config = super(SLU, self).get_config()
        return dict(list(base_config.items()))