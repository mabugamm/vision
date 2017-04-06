# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('image.jpg')
kernel = np.zeros([3,3])
kernel[1,2] = 1
kernel
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121), plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()