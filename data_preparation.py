#Libraries
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt

new_image = plt.imread(img_path)
resized_image = resize(new_image, (256,256,3))
img = plt.imshow(resized_image)

#Resize function - proportionally
new_image = plt.imread(img_path)
resized_image = resize(new_image, (new_image.shape[0] // 6, new_image.shape[1] // 6),
                       anti_aliasing=True)
img= plt.imshow(resized_image)
