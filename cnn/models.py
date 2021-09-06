# Keras Layers
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization

# Utilities
from tensorflow.keras import optimizers


# Binary Classification Model
def binary_classifier(input_dim, feature_detectors, size_feature_detectors, learning_rate):

    classifier = Sequential()

    # Step 1 - Convolution
    # make 32 feature detectors with a size of 3x3
    classifier.add(Conv2D(feature_detectors, size_feature_detectors, input_shape=input_dim, activation="relu"))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a third convolutional layer
    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding an Average Pooling after Convolution
    classifier.add(GlobalAveragePooling2D())

    # Step 4 - Full connection
    classifier.add(Dense(activation="relu", units=128))
    classifier.add(Dense(activation="sigmoid", units=1))

    # Compiling the cnn
    opt = optimizers.Adam(learning_rate=learning_rate)
    classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # Print out the List of Keras Layers
    classifier.summary()

    return classifier


# Multi Class Classification Model
def multi_class_classifier(input_dim, feature_detectors, size_feature_detectors, learning_rate):

    classifier = Sequential()

    # Step 1 - Convolution
    # make 32 feature detectors with a size of 3x3
    classifier.add(Conv2D(feature_detectors, size_feature_detectors, input_shape=input_dim, activation="relu"))

    # Dropout before Pooling
    classifier.add(Dropout(0.2))
    classifier.add(BatchNormalization())

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a third convolutional layer
    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding an Average Pooling after Convolution
    classifier.add(GlobalAveragePooling2D())

    # Step 4 - Full connection
    classifier.add(Dense(activation="relu", units=128))
    classifier.add(Dense(activation="softmax", units=9))

    # Compiling the cnn
    opt = optimizers.Adam(learning_rate=learning_rate)
    classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Print out the List of Keras Layers
    classifier.summary()

    return classifier
