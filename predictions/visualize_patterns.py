"""September 2020 for Forschungsmodul SoSe 20 at the DDU at TU Darmstadt"""

'@author: Paul Steggemann (github@ Paulinos739)'

'This program tests a pretrained DNN Classification model on a floor plan image'
'and visualizes the predicted output through footers which are pasted onto the image'

""

# Keras Preprocessing and Numpy Libraries needed
from keras.models import load_model
from keras.preprocessing import image
from typing import Tuple
import numpy as np

# Create a list to put in the label predictions
label_list_Numpy_2D = []

# path to test image at first!
test_image_path = 'predictions/test.png'

# load test img and convert to Numpy array
test_image = image.load_img(test_image_path,
                            target_size=(64, 64), color_mode="rgb")
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)


# Model to distinguish rectangular-shaped architecture
def classifier_rectangle():
    classifier = load_model(
        'trained model as h5 file',
        compile=True)

    result = classifier.predict(test_image)
    label_list_Numpy_2D.append(result)
    return classifier


# Model to distinguish composite-rectangular shapes
def classifier_longitudinal():
    classifier = load_model(
        'trained model as h5 file',
        compile=True)

    result = classifier.predict(test_image)
    label_list_Numpy_2D.append(result)
    return classifier


# Model to distinguish linear-shaped architecture
def classifier_linear():
    classifier = load_model(
        'trained model as h5 file',
        compile=True)

    result = classifier.predict(test_image)
    label_list_Numpy_2D.append(result)
    return classifier


# Model to distinguish circular-shaped architecture
def classifier_circle():
    classifier = load_model(
        'trained model as h5 file',
        compile=True)

    result = classifier.predict(test_image)
    label_list_Numpy_2D.append(result)
    return classifier


# Model to distinguish polygon-shaped architecture
def classifier_polygon():
    classifier = load_model(
        'trained model as h5 file',
        compile=True)

    result = classifier.predict(test_image)
    label_list_Numpy_2D.append(result)
    return classifier


# Model to distinguish organically-shaped architecture
def classifier_organic():
    classifier = load_model(
        'trained model as h5 file',
        compile=True)

    result = classifier.predict(test_image)
    label_list_Numpy_2D.append(result)
    return classifier


# Model to distinguish if the architecture has an Atrium or not
def classifier_atrium():
    classifier = load_model(
        'trained model as h5 file'
        , compile=True)

    result = classifier.predict(test_image)
    label_list_Numpy_2D.append(result)
    return classifier


# Model to distinguish if the building has a column grid
def classifier_column_grid():
    classifier = load_model(
        'trained model as h5 file',
        compile=True)

    test_image_Column = image.load_img(test_image_path,
                                       target_size=(128, 128, 1), color_mode="grayscale")
    test_image_Column = image.img_to_array(test_image_Column)
    test_image_Column = np.expand_dims(test_image_Column, axis=0)
    result = classifier.predict(test_image_Column)
    label_list_Numpy_2D.append(result)
    return classifier


# Model to distinguish if the plan have staircases
def classifier_staircase():
    classifier = load_model(
        'trained model as h5 file',
        compile=True)

    test_image_Treppe = image.load_img(test_image_path,
                                       target_size=(64, 64, 1), color_mode="grayscale")
    test_image_Treppe = image.img_to_array(test_image_Treppe)
    test_image_Treppe = np.expand_dims(test_image_Treppe, axis=0)
    result = classifier.predict(test_image_Treppe)
    label_list_Numpy_2D.append(result)
    return classifier


def Visualizer():
    # Call the predictions
    classifier_rectangle()
    classifier_longitudinal()
    classifier_linear()

    classifier_circle()
    classifier_polygon()
    classifier_organic()

    classifier_atrium()
    classifier_column_grid()
    classifier_staircase()

    # Convert a 2D Numpy Array to a Python list to parse it
    from numpy import ndarray
    label_list = np.array(label_list_Numpy_2D)
    list(label_list_Numpy_2D)
    ndarray.tolist(label_list)

    print("Printing a python list of estimated patterns...")
    print(label_list)

    # create the visualization
    def add_legend_to_image(image_path: test_image_path,
                            feature_vector: Tuple[int, int, int, int, int, int, int, int, int]):

        from PIL import Image
        from PIL import ImageFont
        from PIL import ImageDraw

        # resize the test image and paste on new canvas for visualization
        img = Image.open(image_path)
        img = img.resize((1600, 1024), Image.ANTIALIAS)
        new_image = Image.new("RGB", (1500, 1400), color="white")
        new_image.paste(img, (0, 0, 1600, 1024))

        draw = ImageDraw.Draw(new_image)
        font = ImageFont.truetype(
            "your favourite typeface as .ttf or equivalent",
            19,
            encoding="unic")

        # List to process the predictions from the classifiers
        labels_to_add = []
        if feature_vector[0]:
            labels_to_add.append(("Rectangle", "black"))
        if feature_vector[1]:
            labels_to_add.append(("Composite", "pink"))
        if feature_vector[2]:
            labels_to_add.append(("Longitudinal", "blue"))
        if feature_vector[3]:
            labels_to_add.append(("Circle", "violet"))
        if feature_vector[4]:
            labels_to_add.append(("Polygonal", "gray"))
        if feature_vector[5]:
            labels_to_add.append(("Organic", "purple"))
        if feature_vector[6]:
            labels_to_add.append(("Atrium", "magenta"))
        if feature_vector[7]:
            labels_to_add.append(("Columns", "Orange"))
        if feature_vector[8]:
            labels_to_add.append(("Staircase", "green"))

        rectangle_x1 = 15
        text_x1 = 20

        if labels_to_add:
            draw.text(xy=(10, 610), text="Architectural Patterns", fill="black", font=font)
            for label in labels_to_add:
                draw.rectangle(xy=(rectangle_x1, 640, rectangle_x1 + 110, 670), fill=label[1])
                draw.text(xy=(text_x1, 645), text=label[0], font=font, fill="white")
                rectangle_x1 += 130
                text_x1 += 130
        else:
            draw.text(xy=(10, 610), text="Did not find any pattern", fill="black")

        new_image.show()
        new_image.save('predicted_1.png')

    # load the image again for display
    add_legend_to_image(image_path=test_image_path, feature_vector=label_list)


# Run Visualizer
if __name__ == '__main__':
    Visualizer()
