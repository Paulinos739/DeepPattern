'Created on Mon Sep  07 10:05:25 2020'
'Modified July 2020 for Forschungsmodul SoSe 20 at the DDU at TU Darmstadt'
'@author: Paul Steggemann (github@ Paulinos739)'

'This binary classification model predicts the presence of a label (Architectural Pattern) in Floor Plan Images. '

""

# Importing the Keras libraries
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Dense, GaussianDropout, BatchNormalization


# Initializing the CNN
def create_classifier_circle():
    # CNN model heavenly borrowed from @author: pranavjain
    # Initialising the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    # make 32 feature detectors with a size of 3x3
    # choose the input-image's format to be 64x64 with 3 channels
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

    classifier.add(GaussianDropout(0.2))
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
    classifier.add(Dense(activation="sigmoid", units=1))

    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier


# Data preprocessing with one directory(with test and validation img), plus a Csv file containing binary labels

def main():
    import itertools
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
            zoom_range=0.1,
            width_shift_range=0.2,
            shear_range=0.2,
            rotation_range=0.3,
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
            csv_file='dataset/data.csv',
            image_directory='dataset',
            batch_size=50,
            image_column_name="Sample",
            class_column_names=["Circle"],  # Set the label to fit to, "Circle" in this case - could be any of all teh label
            validation_split=0.2,
        )
        
        # start computation and train the model
        classifier = create_classifier_circle()  # set which CNN model is used here!
        history = classifier.fit(x=train_data,
                                 steps_per_epoch=None,
                                 epochs=100,
                                 validation_data=validation_data,
                                 validation_steps=None,
                                 )

        # Visualize the performance after training
        def model_performance_plotting():
            print(history.history.keys())

            from matplotlib import pyplot as plt

            # summarize history for accuracy
            plt.style.use("ggplot")
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Algorithm accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'])
            plt.savefig('model_Circle_accuracy.png')

            # summarize history for loss
            plt.style.use("ggplot")
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Algorithm loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'])
            plt.savefig('model_Circle_loss.png')

        # Save trained CNN to a HDF file
        def save_classifier(save_h5=True):
            if save_h5:
                classifier.save(
                    "trained_classifier_circle.h5",
                    overwrite=True, include_optimizer= True
                )
                print("fitted model successfully saved to disk")
            else:
                print("model was not saved to disk")
                pass

        # Training history
        print(history.history.keys())
        # Plot out the performance graphs
        model_performance_plotting()
        # Save fitted model
        save_classifier()


# Run program
if __name__ == '__main__':
    main()
