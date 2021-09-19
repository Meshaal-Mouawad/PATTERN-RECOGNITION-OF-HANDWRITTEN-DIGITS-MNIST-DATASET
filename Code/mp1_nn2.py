"""
    Digital Signal Processing
    Mini Project 1 - Deep Neural Network TensorFlow code

    Adapted from Advanced Deep Learning with TensorFlow 2 and Keras by Rowel Atienza
    https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras

    Requires Tensorflow 2.1 CPU

    John Ball
    13 Aug 2020

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
import numpy as np
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist
from confusion_matrix import confusion_matrix, print_confusion_matrix_stats


# Force code to run on the CPU
tensorflow.config.set_visible_devices([], 'GPU')

with tensorflow.device('/cpu:0'):

    # Load mnist dataset
    print("\nLoading MNIST dataset.")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Compute the number of labels
    num_labels = len(np.unique(y_train))

    # Convert to one-hot vectors
    y_train = to_categorical(y_train, num_labels)
    y_test = to_categorical(y_test, num_labels)

    # Get the image dimensions (assumed square)
    image_size = x_train.shape[1]
    input_size = image_size * image_size

    # Resize and normalize
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Network parameters
    # image is processed as is (square grayscale)
    input_shape = (image_size, image_size, 1)
    batch_size = 128
    kernel_size = 3
    pool_size = 2
    filters = 16
    dropout = 0.2

    # model is a stack of CNN-ReLU-MaxPooling
    model = Sequential()
    model.add(Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size))
    model.add(Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size))
    model.add(Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     activation='relu'))
    model.add(Flatten())

    # dropout added as regularizer
    model.add(Dropout(dropout))

    # output layer is 10-dim one-hot vector
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    # Print and plot model
    model.summary()
    plot_model(model, to_file='mp1_nn1.png', show_shapes=True)

    # loss function for one-hot vector
    # use of adam optimizer
    # accuracy is good metric for classification tasks
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # train the network
    model.fit(x_train, y_train, epochs=2, batch_size=batch_size)

    # Compute predictions (test mode) for training data
    y_pred = model.predict(x_train,
                           batch_size=batch_size,
                           verbose=0)

    # Convert one-hot tensors back to class labels (and make them numpy arrays)
    y_train_true_class = K.argmax(y_train).numpy()
    y_train_pred_class = K.argmax(y_pred).numpy()

    # create CM and print confusion matrix and stats
    cm_train, cm_train_acc = confusion_matrix(y_train_true_class, y_train_pred_class)
    print_confusion_matrix_stats(cm_train, 'Train')

    # Validate the model on test dataset to determine generalization
    y_pred = model.predict(x_test,
                           batch_size=batch_size,
                           verbose=0)

    # Convert one-hot tensors back to class labels (and make them numpy arrays)
    y_test_true_class = K.argmax(y_test).numpy()
    y_test_pred_class = K.argmax(y_pred).numpy()

    # create CM and print confusion matrix and stats
    cm_test, cm_test_acc = confusion_matrix(y_train_true_class, y_train_pred_class)
    print_confusion_matrix_stats(cm_test, 'Test')
