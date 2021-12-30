import keras
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Flatten, Dense,Dropout, Softmax, ReLU

@tf.keras.utils.register_keras_serializable()
class SquarePlus(Layer):
    def __init__(self, bParameter, name, **kwargs):
        super(SquarePlus, self).__init__(name=name)
        self.bParameter = bParameter
        super(SquarePlus, self).__init__(**kwargs)

    def build(self, input_shape):
        self.SbParameter = tf.constant(self.bParameter, dtype=tf.float32)
        self.SPointFive = tf.constant(0.5, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = keras.backend.pow(inputs, 2) + self.SbParameter
        x = (keras.backend.sqrt(x) + inputs) * self.SPointFive
        return x

    def get_config(self):
        base_config = super(SquarePlus, self).get_config()
        config = {
            "bParameter": self.bParameter,
        }
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable()
class AdaptSquarePlus(Layer):
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


NUM_CLASSES = 10
BatchSize = 128
save_train_path = r"F:\练习模型训练权重\SquarePlus"
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=r"C:\Users\samzhang\.keras\datasets\mnist.npz")

# expand the channel dimension
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# make the value of pixels from [0, 255] to [0, 1] for further process
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# convert class vectors to binary class matrics
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

model = tf.keras.Sequential()
model.add(Conv2D(filters=6, kernel_size=5, padding="SAME", kernel_regularizer=l2(1e-5), input_shape=(28, 28, 1)))
# model.add(SquarePlus(bParameter=4, name="SquarePlus_1"))
model.add(AdaptSquarePlus(name="SquarePlus_1"))
# model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=16, kernel_size=5, padding="SAME", kernel_regularizer=l2(1e-5)))
model.add(AdaptSquarePlus(name="SquarePlus_2"))
# model.add(SquarePlus(bParameter=4, name="SquarePlus_2"))
# model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=120, kernel_size=5, padding="SAME", kernel_regularizer=l2(1e-5)))
model.add(AdaptSquarePlus(name="SquarePlus_3"))
# model.add(SquarePlus(bParameter=4, name="SquarePlus_3"))
# model.add(ReLU())

model.add(Flatten())
model.add(Dense(units=84))
model.add(Dropout(rate=0.5))

model.add(Dense(units=10))
model.add(Softmax())

model.build((BatchSize, 28,  28, 1))
model.summary()

kept = tf.keras.callbacks.ModelCheckpoint(save_train_path + "/" + "model.{epoch:02d}-{val_loss:.4f}-" +
                                          ".h5", monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='max')

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=BatchSize, epochs =300, verbose=1, validation_data=(x_test, y_test),
          callbacks=[kept])
score = model.evaluate(x_test, y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


