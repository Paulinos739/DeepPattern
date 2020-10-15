"""September 2020 for Forschungsmodul SoSe 20 at the DDU at TU Darmstadt"""

'@author: Paul Steggemann (github@ Paulinos739)'
'Code partly borrowed from author: github @ritiek'

'This program tests a pretrained Classification DNN model on a number of floor plan images. '
'Images to test the model on. All have to be located in the same folder'

""

# Import Libraries
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image


# Model to distinguish with multi-label classification

def classifier_multi_label():
    classifier = load_model(
        r'C:\Users\PAUL\PycharmProjects\FM_SoSo20_Master\trained_classifiers_weights\multi_label\Multi_label_100_CNN.h5',
        compile=True)

    return classifier


def main():

    def floor_plan_prediction_multiple():
        # image folder
        folder_path = 'path to folder containing the test data'

        # dimensions of images
        img_width, img_height = 64, 64

        # load the trained model
        model = classifier_multi_label()

        # put all images into a list
        images = []
        for img in os.listdir(folder_path):
            img = os.path.join(folder_path, img)
            img = image.load_img(img, target_size=(img_width, img_height))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            images.append(img)

        # stack up images list to pass for prediction
        images = np.vstack(images)
        result = model.predict(images, batch_size=None, verbose=1)  # or predict_classes when using only one model
        print(result)

    # Call the function
    floor_plan_prediction_multiple()


# Run Visualizer
if __name__ == '__main__':
    main()
