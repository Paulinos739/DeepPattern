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
    classifier = load_model('path/to/model',
        compile=True)
    return classifier


 def floor_plan_prediction_multiple():
        # image folder
        folder_path = 'path/to/folder/containing/test/data'

        # put all images into a list and...
        images = []
        for img in os.listdir(folder_path):
            img = os.path.join(folder_path, img)
            img = image.load_img(img, target_size=(64, 64))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            images.append(img)

        # ...stack up images list to pass for prediction
        images = np.vstack(images)

        # Run Prediction
        result = classifier_multi_label().predict(images)
        print(result)


if __name__ == '__main__':
    floor_plan_prediction_multiple()
