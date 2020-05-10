import numpy as np
import cv2
import imutils
from skimage.feature import hog
import matplotlib.pyplot as plt




im = cv2.imread("./images/book_01008.jpg")
#im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
im = cv2.resize(im, (200,200))


fd, HI = hog(im, orientations=12, pixels_per_cell=(4,4),
                            cells_per_block=(1,1),block_norm='L1', visualize=True, 
                            transform_sqrt=True, feature_vector=True, multichannel=True) 

fd, HI2 = hog(im, orientations=9, pixels_per_cell=(8,8),
                            cells_per_block=(2,2),block_norm='L2', visualize=True, 
                            transform_sqrt=True, feature_vector=True, multichannel=True) 

numpy_horizontal = np.hstack((HI, HI2))
plt.imshow(numpy_horizontal)
k = cv2.waitKey(1000)
