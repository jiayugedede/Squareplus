import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense,Dropout, Softmax, ReLU

# Trainable activation function with differentiable negative side and adaptable rectified point
@tf.keras.utils.register_keras_serializable()
class LPSELU(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        super(LPSELU, self).__init__(name=name)
        super(LPSELU, self).__init__(**kwargs)

    def build(self, input_shape):
        b, w, h, c = input_shape
        self.initializer = tf.constant_initializer(value=1.6733)
        self.alpha = tf.Variable(initial_value=self.initializer(shape=[c], dtype=tf.float32),
                                 dtype=tf.float32, trainable=True, name="alpha")

        self.s_initializer = tf.constant_initializer(value=1.0507)
        self.s_lambda = tf.Variable(initial_value=self.s_initializer(shape=[c], dtype=tf.float32),
                                    dtype=tf.float32, trainable=True, name="lambda")

        self.a_initializer = tf.constant_initializer(value=0.01)
        self.a = tf.Variable(initial_value=self.a_initializer(shape=[c], dtype=tf.float32),
                                    dtype=tf.float32, trainable=True, name="a")
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        pos = tf.nn.relu(inputs)
        lambda_pos = self.s_lambda * pos
        neg = (inputs - tf.math.abs(inputs)) * 0.5
        lambda_neg = (self.alpha * tf.math.exp(neg) - self.alpha) * self.s_lambda + self.a * neg
        return lambda_neg + lambda_pos

    def get_config(self):
        base_config = super(LPSELU, self).get_config()
        return dict(list(base_config.items()))
