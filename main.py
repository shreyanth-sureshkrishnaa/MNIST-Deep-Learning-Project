# Imports

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist # A dataset of handwritten digits from 0 - 9. 28x28 grayscale images.
from tensorflow.keras.models import Sequential # In this model, data flows in one direction. It's a linear stack of layers.
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical # Needed for categorical_crossentropy




def main():

    # Setting a seed ensures approximately the same results across different runs.
    seed = 7
    np.random.seed(seed)


    # X_train has 60,000 images. X_test has 10,000.
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()


    # Data preprocessing:

    # Normalize and reshape
    # We're diviing by 255 here to normalize the original pixel values to the 0.0 - 1.0 range.
    # This stabilizes gradients, speeds up conversions. etc.

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    num_pixels = X_train.shape[1] * X_train.shape[2] # Each image becomes a 784-dimensional vector ( 28 x 28 )

    # Manual flattening.
    X_train = X_train.reshape(X_train.shape[0], num_pixels)
    X_test = X_test.reshape(X_test.shape[0], num_pixels)


    num_classes = 10
    Y_train = to_categorical(Y_train, num_classes)
    Y_test = to_categorical(Y_test, num_classes)

    # Model architecture:
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(num_pixels,)))
    model.add(Dropout(0.2)) # Randomly disables 20% of neurons during training to prevent overfitting.
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1]*100))


if __name__ == "__main__":
    main()


