# DeepPattern

## Introducing Convolutional Neural Networks for Architectural Pattern Recognition
![Floor_PLan_Grid_highRes](https://user-images.githubusercontent.com/65179419/96243449-0dc58100-0fa5-11eb-9ab1-b1160f38fbf8.png)

Independent Research Project Spring 2020 @[Digital Design Unit](https://ddu-research.com/) at TU Darmstadt


### Abstract
The promising methodology deriving from Christopher Alexanders design-concept of Pattern Languages needs to be revisited and approached on a more holistic level. This paper proposes to leverage the potential of Big Data and Machine Vision to analyticity categorize and retrieve patterns in architectural floor plans.
The research uses Convolutional Neural Networks to automatically detect design patterns across varying typologies, centuries and styles. The title DeepPattern implies the connection between the programming task of Pattern Recognition with Deep Learning and the theory of Pattern Languages. The method proposed by Alexander et al is directly applied to the task of extracting data in architecture with digital methods, for which I propose the term Architectural Pattern Recognition.
The conducted experiment proved that the algorithms are capable of understanding basic spatial concepts of architectural design, obtaining results from 80% up to 98% accuracy during training on nine architectural design categories in total, trained only on a small dataset of public buildings.
Finally, this paper also presents a simple tool which visualizes the automated prediction of patterns on floor plans through the trained Machine Vision Bots.


### Retrieving Architectural Patterns with CNN

The framework developed here can be used to automatically classify floor plans acoording to its patterns. Use the tool, put a floor plan image in and test it out on nine different architectural patterns, which gets then visualized in the way demonstrated below. It is part of an ongoing investigation how it is possible to integrate Machine Learning into digital yet human-centered architectural design processes.

![predict_11](https://user-images.githubusercontent.com/65179419/96238135-6e9d8b00-0f9e-11eb-8f6e-0edec565875f.png)


### Browse architectural history

![Suggestive Architecture_new](https://user-images.githubusercontent.com/65179419/96238321-a4427400-0f9e-11eb-9824-5b456f9cab71.png)


### Usage

This repo contains pure python scripts which can be used to learn and classify architectural patterns in floor plan images with Convolutional Neural Networks. Pattern Recognition in architectural design may not be confused with Pattern Recogntion in Data Science. For furter reading, see: [Christopher Alexander. A Pattern Language, Berkeley 1977](https://en.wikipedia.org/wiki/A_Pattern_Language)
In total nine bots were set up for to learn each of nine labels which frequently occur in architecture: Rectangle, Composite, Circle, longitudinal, Polygonal, Organic, Column Grid and Staircase(check Icons below).

I used a simple Sequential model. For data-preprocessing, make sure to put all your images in one folder and to make a csv-file in which all labels of the samples exist. With the flow_from_dataframe method, a keras built-in function, the data is preprocessed and augmented. Make sure to stick to this structure of the directory.
You can use binary classification and train each model only on one label/pattern at the same time, or it is possible to also use multi-label classification with the code demonstrated here.

**Dependencies:** It uses the [Keras](https://keras.io/) Library for the Neural Networks and runs on [tensorflow](https://www.tensorflow.org/) backend. So make sure to have both installed on your machine.
