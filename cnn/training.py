' July 2020 for Forschungsmodul SoSe 20 at the DDU at TU Darmstadt'
'@author: Paul Steggemann (github@ Paulinos739)'

'This binary classification model predicts the presence of a label (Arhitectural Patterns) in Floor Plan Images. '
""

# Libraries for all
from tensorflow import keras
import os
from typing import Union, List
import pandas as pd

# only defined for semantic distinction, really and it will be str in most cases
Path = Union[str, os.PathLike]


def csv_dataset(csv_file: Path, image_directory: Path, batch_size: int, image_column_name: str,
                class_column_names: List[str], validation_split: float):

    dataframe = pd.read_csv(csv_file)

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.1,
        width_shift_range=0.2,
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


def main():
    train_data, validation_data = csv_dataset(
        csv_file="",
        image_directory="",
        batch_size=50,
        image_column_name="Sample",
        class_column_names=["class A"],  # Set the category to fit to
        validation_split=0.2,
    )

    # Set up training parameters
    initial_learning_rate = 0.01
    epochs = 600

    # start training / fit the model / save results
    from models import binary_classifier
    from models import multi_class_classifier
    from training_utilities import training_callbacks, model_performance_plotting, save_classifier

    # Set model architecture
    classifier = binary_classifier(input_dim=(64, 64, 3), feature_detectors=32, size_feature_detectors=(4, 4),
                                   learning_rate=initial_learning_rate)
    history = classifier.fit(x=train_data,
                             steps_per_epoch=None,
                             epochs=epochs,
                             validation_data=validation_data,
                             validation_steps=None,
                             callbacks=training_callbacks(ProgbarLogger=False, TensorBoard=True, CSVLogger=True,
                                                          ModelCheckpoint=True, LearningRateScheduler=True))

    model_performance_plotting(record=history)
    save_classifier(fitted_model=classifier, format="tf_SavedModel")


if __name__ == '__main__':
    main()
