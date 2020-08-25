#Libraries
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt

#Resize the floor plan images 
new_image = plt.imread(img_path)
resized_image = resize(new_image, (256,256,3))
img = plt.imshow(resized_image)

#Resize function - but proportionally
new_image = plt.imread(img_path)
resized_image = resize(new_image, (new_image.shape[0] // 6, new_image.shape[1] // 6),   ##Which resolution does this resize function otput?
                       anti_aliasing=True)
img= plt.imshow(resized_image)    
