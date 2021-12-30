import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


@tf.keras.utils.register_keras_serializable()
class AdaptSquarePlus(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        super(AdaptSquarePlus, self).__init__(name=name)
        super(AdaptSquarePlus, self).__init__(**kwargs)

    def build(self, input_shape):
        self.SbParameter = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True)
        self.SPointFive = tf.constant(0.5, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs,  *args, **kwargs):
        x = keras.backend.pow(inputs, 2) + self.SbParameter
        x = (keras.backend.sqrt(x) + inputs) * self.SPointFive
        return x

    def get_config(self):
        base_config = super(AdaptSquarePlus, self).get_config()
        return dict(list(base_config.items()) )


@tf.keras.utils.register_keras_serializable()
class SquarePlus(tf.keras.layers.Layer):
    def __init__(self, bParameter, name, **kwargs):
        super(SquarePlus, self).__init__(name=name)
        self.bParameter = bParameter
        self.PointFive = 0.5
        super(SquarePlus, self).__init__(**kwargs)

    def build(self, input_shape):
        self.SbParameter = tf.constant(self.bParameter, dtype=tf.float32)
        self.SPointFive = tf.constant(self.PointFive, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = keras.backend.pow(inputs, 2) + self.bParameter
        x = (keras.backend.sqrt(x) + inputs) * self.PointFive
        return x

    def get_config(self):
        base_config = super(SquarePlus, self).get_config()
        config = {
            "bParameter": self.bParameter,
            "PointFive": self.PointFive,
        }
        return dict(list(base_config.items()) + list(config.items()))

def load_trained_model():
    _custom_objects = {
        "Custom>SquarePlus" : SquarePlus
        }
    model_name = r"F:\练习模型训练权重\SquarePlus\model.14-0.0545-.h5"
    function_model = load_model(model_name, custom_objects=_custom_objects)
    # function_model = load_model(model_name)
    print('model load success')
    return function_model


def loadMnistTest():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=r"C:\Users\samzhang\.keras\datasets\mnist.npz")
    # expand the channel dimension
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # make the value of pixels from [0, 255] to [0, 1] for further process
    x_test = x_test.astype('float32') / 255.
    return x_test, y_test


def main():
    x_test, y_test = loadMnistTest()
    model = load_trained_model()
    score = model.predict(x_test)
    predicts = np.argmax(score, axis=1)

    correct = 0
    for index in  range(len(predicts)):
        if predicts[index]==y_test[index]:
            correct +=1
    correct_rate = correct / len(y_test)
    print(correct_rate)

if (__name__ == "__main__"):
    main()
