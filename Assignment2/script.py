import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("normal.jpg")
img_g = np.zeros( (img.shape[0], img.shape[1]), dtype=np.uint8 )
edges = cv2.Canny(img_g,60, 100, L2gradient=True);
plt.subplot(121), plt.imshow(img_g)
plt.subplot(122), plt.imshow(edges)
plt.show()