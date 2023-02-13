import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense,Dropout, Softmax, ReLU

# Not UTF-8, GBK is ok.
#SMU: SMOOTH ACTIVATION FUNCTION FOR DEEP NETWORKS USING SMOOTHING MAXIMUM TECHNIQUE
# This is SUM, not SUM-1 activation method.
# classification: alpha=0.25, mu=1.0.
# object detection: alpha=0.01, mu=2.5.
# alpha and mu can be either hyperparameters or trainable parameters.
@tf.keras.utils.register_keras_serializable()
class SMU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.25, name="SUM", **kwargs):
        super(SMU, self).__init__(name=name)
        self.alpha = alpha
        super(SMU, self).__init__(**kwargs)

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.initializer = tf.constant_initializer(value=1.0)
        self.MU = tf.Variable(initial_value=self.initializer(shape=[c], dtype=tf.float32),
                                       dtype=tf.float32,
                                       trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        self.forepart = (1 + self.alpha) * inputs
        self.backpart = (1 - self.alpha) * inputs * tf.math.erf(self.MU * (1 - self.alpha) * inputs)
        self.result = (self.forepart + self.backpart) / 2
        return self.result

    def get_config(self):
        config = {"alpha": self.alpha}
        base_config = super(SMU, self).get_config()
        return dict(list(base_config.items())+ list(config.items()))
