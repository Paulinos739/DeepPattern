# DeepPattern

## Introducing Convolutional Neural Networks for Architectural Pattern Recognition

![Cover_Design_A5_machine_view_2page](https://user-images.githubusercontent.com/65179419/96243957-ba9ffe00-0fa5-11eb-8f41-482d4b52a472.png)




Independent Research Project @[Digital Design Unit](https://ddu-research.com/) | TU Darmstadt


### Abstract
The promising methodology deriving from [Christopher Alexanders](https://en.wikipedia.org/wiki/Christopher_Alexander) design-concept of Pattern Languages needs to be revisited and approached on a more holistic level. This paper proposes to leverage the potential of Big Data and Machine Vision to analyticity categorize and retrieve patterns in architectural floor plans.
The research uses Convolutional Neural Networks to automatically detect design patterns across varying typologies, centuries and styles. The title DeepPattern implies the connection between the programming task of Pattern Recognition with Deep Learning and the theory of Pattern Languages. The method proposed by Alexander et al is directly applied to the task of extracting data in architecture with digital methods, for which I propose the term Architectural Pattern Recognition.
The conducted experiment proved that the algorithms are capable of understanding basic spatial concepts of architectural design, obtaining results from 80% up to 98% accuracy during training on nine architectural design categories in total, trained only on a small dataset of public buildings.
Finally, this paper also presents a simple tool which visualizes the automated prediction of patterns on floor plans through the trained Machine Vision Bots.


### Retrieving Architectural Patterns with CNN

The framework developed here can be used to automatically classify floor plans acoording to its patterns. Use the tool, put a floor plan image in and test it out on nine different architectural patterns, which gets then visualized in the way demonstrated below. It is part of an ongoing investigation how it is possible to integrate Machine Learning into digital yet human-centered design processes in the early-concept stages of architecture.

<img src="https://user-images.githubusercontent.com/65179419/96464408-11b50580-1228-11eb-928e-1501e616333d.gif" width="860" height="500"/>




### Browse architectural history

![Suggestive Architecture_new](https://user-images.githubusercontent.com/65179419/96238321-a4427400-0f9e-11eb-9824-5b456f9cab71.png)


### Usage

This repo contains pure python scripts which can be used to learn and classify architectural patterns in floor plan images with Convolutional Neural Networks. Pattern Recognition in architectural design may not be confused with Pattern Recogntion in Data Science. For furter reading, see: [Christopher Alexander. A Pattern Language, Berkeley 1977](https://en.wikipedia.org/wiki/A_Pattern_Language)
In total nine bots were set up for to learn each of nine labels which frequently occur in architecture: Rectangle, Composite, Circle, longitudinal, Polygonal, Organic, Column Grid and Staircase(check Icons below).

I used a simple Sequential model. For data-preprocessing, make sure to put all your images in one folder and to make a csv-file in which all labels of the samples exist. With the flow_from_dataframe method, a keras built-in function, the data is preprocessed and augmented. Make sure to stick to this structure of the directory.
You can use binary classification and train each model only on one label/pattern at the same time, or it is possible to also use multi-label classification with the code demonstrated here. It uses the convenient [Keras](https://keras.io/) Library for the Neural Networks and runs on [tensorflow](https://www.tensorflow.org/install) backend. So make sure to have both installed on your machine.

**Dependencies:** 
- Tensorflow
- Keras
- Pandas
- Matplotlib
- PIL (Pillow)


## Model architecture

For the Neural Network architecture I always the architecture illustrated below and rearranged it slightly depending on which label have been trained. It uses two Convolutional Layers with 2x2 Pooling each. For Pooling I used Global Average Pooling 2D, because architectural features in floor plans occur not only in small regions of the image, but on a broader scale, e.g. the shape of a building occupies the whole image.
Feel free to play with the layers and esspeccially with the input size. This should have a positive impact on the feature learning.

![DNN_diagramm](https://user-images.githubusercontent.com/65179419/96249838-18d0df00-0fae-11eb-8679-edd5449765a4.png)


### Concept video to illustrate results

<a href="http://www.youtube.com/watch?feature=player_embedded&v=b9f7d2NJ6pQ" target="_blank">
 <img src="http://img.youtube.com/vi/b9f7d2NJ6pQ/0.jpg" alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" />
</a>


### References

Aggarwal C. : Neural Networks and Deep Learning, Springer 2018

Alexander, C. (1977) A Pattern Language. Towns, Buildings, Constructions. New York Oxford University Press.

Chaillou, S. Suggestive CAD, Assisting Design through Machine Learning. 2018

Dodge, Samuel & Xu, Jiu & Stenger, Bjorn. (2017). Parsing floor plan images. 358-361. 10.23919/MVA.2017.7986875.

Wiki as Pattern Language. Cunningham, Mehaffy, 2013

Aerial Futures, Certain Measures 2019:
https://certainmeasures.com/aerosphere.html



### Queries

For questions or further ideas, just email me at paul.arch@web.de
