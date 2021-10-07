import os
import sys
import numpy
from PIL import Image
sys.path.insert(0, '..')
from segmentation import digit_segmentation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0
model = keras.Sequential(
    [
        layers.Dense(384, activation='relu'),
        layers.Dense(192, activation='relu'),
        layers.Dense(96, activation='relu'),
        layers.Dense(48, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(10),
    ]
)
model.compile(
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.002),
    metrics=["accuracy"],
)
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
numpy.set_printoptions(linewidth=numpy.inf)
image = Image.open('input.png')
for index, digit in enumerate(digit_segmentation(image)):
    verdicts = model.predict(tf.constant(numpy.array(digit).reshape(1, 784).astype("float32") / 255.0))
    print(numpy.argmax(verdicts), '-----', verdicts)
