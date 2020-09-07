'Created on Mon Sep  07 10:05:25 2020'

'Modified July 2020 for Forschungsmodul SoSe 20 at the DDU at TU Darmstadt'
'@author: Paul Steggemann (github@ Paulinos739)'

'This program classifies grayscale Floor PLan Images into classes.'

""

# Importing the Keras libraries and other packages
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Bring GPU to life
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# Initializing the CNN model (06.09.2020 with Frederik)

def create_classifier(n_layers=5, global_pooling=True):
    # Initialising the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    # make 32 feature detectors with a size of 3x3
    # choose the input-image's format to be 64x64 with 3 channels

    features = 32
    for n in range(n_layers):
        # Convolution
        classifier.add(Conv2D(features, (3, 3), input_shape=(64, 64, 3), activation="relu"))
        # Step 2 - Pooling
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        features *= 2

    # Step 3 - Flattening
    if global_pooling:
        classifier.add(GlobalAveragePooling2D())
    else:
        classifier.add(Flatten())
        classifier.add(Dense(activation="relu", units=128))

    # Step 4 - Full connection
    classifier.add(Dense(activation="sigmoid", units=1))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Print out a list of the CNN-Architecture
    classifier.summary()

    return classifier



# CNN model from @author: pranavjain

def create_classifier_2():
    # Initialising the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    # make 32 feature detectors with a size of 3x3
    # choose the input-image's format to be 64x64 with 3 channels
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(activation="relu", units=128))
    classifier.add(Dense(activation="sigmoid", units=2))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


    # Print out the List of Keras Layers
    classifier.summary()

    return classifier




def main():
    classifier = create_classifier_2()

    # use ImageDataGenerator to preprocess the data
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing import image_dataset_from_directory

    # augment the data that we have
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       horizontal_flip = True,
                                       vertical_flip=True, rotation_range= 30)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    # prepare training data

    train_ds = train_datagen.flow_from_directory(r'C:\Users\PAUL\PycharmProjects\FM_SoSo20_Master\dataset\training_data',
                                                batch_size = 20, target_size= (64,64))

    validation_ds = test_datagen.flow_from_directory(r'C:\Users\PAUL\PycharmProjects\FM_SoSo20_Master\dataset\test_data',
                                                batch_size = 20, target_size= (64,64))



    # start computation and train the model
    classifier.fit(train_ds,
                             steps_per_epoch = (8000/20),
                             epochs = 10,
                             validation_data = validation_ds,
                             validation_steps = 1000/20,
                             )



    # to make predictions
    import numpy as np
    from keras.preprocessing import image
    from IPython.display import display, Image
    from PIL import Image

    test_image = image.load_img(r'C:\Users\PAUL\PycharmProjects\FM_SoSo20_Master\test.png', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)

    if result[0][0] == 1:
        prediction = 'Circle'
    else:
        prediction = 'Cube'


    # Save trained model to a HDF file
    classifier.save(r"C:\Users\PAUL\PycharmProjects\FM_SoSo20_Master\saved_models\circle_cube_trained_CNN.h5",
                    overwrite=True
                    )
    print("trained model succesfully saved to disk")
    print("training finished")



# Run program

if __name__ == '__main__':
    main()
