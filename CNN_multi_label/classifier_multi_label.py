'Created on Mon Sep  07 10:05:25 2020'

'Modified July 2020 for Forschungsmodul SoSe 20 at the DDU at TU Darmstadt'
'@author: Paul Steggemann (github@ Paulinos739)'

'This multi-label classification model learns Architectural Patterns in grayscale Floor Plan Images. '

""

# Importing the Keras libraries
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, BatchNormalization


# Initializing the CNN
def create_classifier():
    # CNN model partly borrowed from @author: pranavjain
    classifier = Sequential()

    # Step 1 - Convolution
    # make 32 feature detectors with a size of 3x3
    # choose the input-image's format to be 64x64 with 3 channels
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

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
    classifier.add(Dense(activation="relu", units=256))
    classifier.add(Dense(activation="relu", units=128))
    classifier.add(Dense(activation="softmax", units=9))

    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print out the List of Keras Layers
    classifier.summary()

    return classifier


# Data preprocessing with one directory(with test and validation img)
def main():
    import os
    from typing import Union, List
    import pandas as pd

    # only defined for semantic distinction, really and will be str in most cases
    Path = Union[str, os.PathLike]

    def csv_dataset(csv_file: Path,
                    image_directory: Path,
                    batch_size: int,
                    image_column_name: str,
                    class_column_names: List[str],
                    validation_split: float = 0.2):

        dataframe = pd.read_csv(csv_file)

        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.2,
            zoom_range=0.1,
            validation_split=validation_split,
        )

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=image_directory,
            x_col=image_column_name,
            y_col=class_column_names,
            class_mode="raw",
            batch_size=batch_size,
            subset="training",
            shuffle=True,
            target_size=(64, 64)
        )

        validation_generator = train_datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=image_directory,
            x_col=image_column_name,
            y_col=class_column_names,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=True,
            subset="validation",
            target_size=(64, 64)
        )

        return train_generator, validation_generator

    if __name__ == '__main__':
        train_data, validation_data = csv_dataset(
            csv_file="set path to CSV-file which contains the label of samples",
            image_directory="set the path to folder where all images are located",
            batch_size=50,
            image_column_name="Sample",  # Column containing the image names
            class_column_names=["Kreis", "L", "Rechteck", "linear", "Polygon", "organisch", "Hof", "Treppe",
                                "Stuetzenraster"]
        )

        # start computation and train the model
        classifier = create_classifier()
        history = classifier.fit(x=train_data,
                                 steps_per_epoch=None,  # max is number of samples/ batch-size
                                 epochs=100,
                                 validation_data=validation_data,
                                 validation_steps=None,  # max is number of samples/ batch-size
                                 )

        # Visualize the performance after training
        def model_performance_plotting():
            print(history.history.keys())

            from matplotlib import pyplot as plt

            # summarize history for accuracy
            plt.style.use("ggplot")
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'])
            plt.savefig('model_multi_accuracy.png')

            # summarize history for loss
            plt.style.use("ggplot")
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'])
            plt.savefig('model_multi_loss.png')

        # Save trained CNN to a HDF file
        def save_classifier(save_h5=True):
            if save_h5:
                classifier.save("trained_classifier.h5",
                                overwrite=True, include_optimizer=True
                                )
                print("Model successfully saved to disk")
            else:
                print("Model was not saved to disk")

        # Training history
        print(history.history.keys())
        # Plot out the performance graphs
        model_performance_plotting()
        # Save fitted model
        save_classifier()


if __name__ == '__main__':
    main()
