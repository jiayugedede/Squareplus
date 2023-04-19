import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self, name="HardSigmoid", **kwargs):
        super(HardSigmoid, self).__init__(name=name)
        self.relu6 = tf.keras.layers.ReLU(max_value=6, name="ReLU6", **kwargs)
        super(HardSigmoid, self).__init__(**kwargs)

    def call(self, input):
        return self.relu6(input + 3.0) / 6.0

    def get_config(self):
        base_config = super(HardSigmoid, self).get_config()
        return dict(list(base_config.items()))

