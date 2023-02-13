import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense,Dropout, Softmax, ReLU

#RSigELU: A nonlinear activation function for deep neural networks
@tf.keras.utils.register_keras_serializable()
class RSigELUD(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        super(RSigELUD, self).__init__(name=name)
        super(RSigELUD, self).__init__(**kwargs)

    def build(self, input_shape):
        b, w, h, c = input_shape
        self.initializer = tf.constant_initializer(value=0.5)
        self.alpha = tf.Variable(initial_value=self.initializer(shape=[c], dtype=tf.float32),
                                       dtype=tf.float32,
                                       trainable=True, name="alpha")

        self.beta_initializer = tf.constant_initializer(value=0.5)
        self.beta = tf.Variable(initial_value=self.beta_initializer(shape=[c], dtype=tf.float32),
                                dtype=tf.float32,
                                trainable=True, name="beta")

        super().build(input_shape)

    def computing_upper_one(self, x, coefficient):
        y = x * (1/(1 + tf.math.exp(-x)))*coefficient + x
        return y

    def computing_neg(self, x, beta_coefficient):
        y = beta_coefficient * (tf.math.exp(x)-1)
        return y

    def call(self, inputs, *args, **kwargs):
        pos = tf.nn.relu(inputs)
        upper_one = tf.math.maximum(1.0, inputs)
        neg = (inputs - tf.math.abs(inputs)) * 0.5
        middle = pos - upper_one

        upper_result = self.computing_upper_one(upper_one, self.alpha)
        less_result = self.computing_neg(neg, self.beta)
        result = upper_result + middle + less_result
        return result

    def get_config(self):
        base_config = super(RSigELUD, self).get_config()
        return dict(list(base_config.items()))


#RSigELU: A nonlinear activation function for deep neural networks
@tf.keras.utils.register_keras_serializable()
class RSigELUS(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        super(RSigELUS, self).__init__(name=name)
        super(RSigELUS, self).__init__(**kwargs)

    def build(self, input_shape):
        b, w, h, c = input_shape
        self.initializer = tf.constant_initializer(value=0.5)
        self.alpha = tf.Variable(initial_value=self.initializer(shape=[c], dtype=tf.float32),
                                       dtype=tf.float32,
                                       trainable=True)
        super().build(input_shape)

    def computing_upper_one(self, x, coefficient):
        y = x * (1/(1 + tf.math.exp(-x)))*coefficient + x
        return y

    def computing_neg(self, x, coefficient):
        y = coefficient * (tf.math.exp(x)-1)
        return y

    def call(self, inputs, *args, **kwargs):
        pos = tf.nn.relu(inputs)
        upper_one = tf.math.maximum(1.0, inputs)
        neg = (inputs - tf.math.abs(inputs)) * 0.5
        middle = pos - upper_one

        upper_result = self.computing_upper_one(upper_one, self.alpha)
        less_result = self.computing_neg(neg, self.alpha)
        result = upper_result + middle + less_result
        return result

    def get_config(self):
        base_config = super(RSigELUS, self).get_config()
        return dict(list(base_config.items()))

