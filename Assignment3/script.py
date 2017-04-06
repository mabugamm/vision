import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('Model barack.png')
cv2.imshow('barack', img)
cv2.waitKey(0)