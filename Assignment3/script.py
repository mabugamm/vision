import cv2
import numpy as np
from matplotlib import pyplot as plt
#img = cv2.imread('Model barack.png')
#cv2.imshow('barack', img)
#cv2.waitKey(0)
a = np.array([[1,2,1],[3,4,2]])
mu = np.average(a, axis=0)
print(mu)
print(a-mu)
print(mu)
print(a)
b = np.array([1,1,1])
print(a.transpose())
print(b.shape[0])
print(np.dot(b,a.transpose()))
print(range(0,4))
print(np.zeros(10))