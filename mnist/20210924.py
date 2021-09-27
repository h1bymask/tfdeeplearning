import os
import numpy
import PIL.ImageOps    
from PIL import Image
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
        layers.Dense(192, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10),
    ]
)
model.compile(
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.01),
    metrics=["accuracy"],
)
model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
for file in ('5-1.png', '5-2.png', '5-3.png'):  # FYI: x_test[0] это тоже цифра 5 от руки
    #PIL.ImageOps.invert(Image.open)
    print(file, model.predict(tf.constant(numpy.array(Image.open(file).convert('L')).reshape(1, 784).astype("float32") / 255.0)))
