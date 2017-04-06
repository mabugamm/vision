'''
Cell counting.

'''

import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def detect(img):
    '''
    Do the detection.
    '''
    #create a gray scale version of the image, with as type an unsigned 8bit integer
    img_g = np.zeros( (img.shape[0], img.shape[1]), dtype=np.uint8 )
    img_g[:,:] = img[:,:,0]

    #1. Do canny (determine the right parameters) on the gray scale image
    edges = cv2.Canny(img_g, 45, 90, L2gradient=True)
    
    #Show the results of canny
    canny_result = np.copy(img_g)
    canny_result[edges.astype(np.bool)]=0
    cv2.imshow('img',canny_result)
    cv2.waitKey(0)

    #2. Do hough transform on the gray scale image
    circles = cv2.HoughCircles(img_g, cv.CV_HOUGH_GRADIENT, dp=4, minDist=25, param1=100, param2=80, minRadius= 20, maxRadius=70)
    circles = circles[0,:,:]
    
    #Show hough transform result
    showCircles(img, circles)
    
    #3.a Get a feature vector (the average color) for each circle
    nbCircles = circles.shape[0]
    features = np.zeros( (nbCircles,3), dtype=np.int)
    for i in range(nbCircles):
        features[i,:] = getAverageColorInCircle(img, int(circles[i,0]), int(circles[i,1]), int(circles[i,2]) )
    
    #3.b Show the image with the features (just to provide some help with selecting the parameters)
    showCircles(img, circles, [ str(features[i,:]) for i in range(nbCircles)] )

    #3.c Remove circles based on the features
    selectedCircles = np.zeros( (nbCircles), np.bool)
    std = np.std(features,axis=1)
    avg = np.average(features, axis=1)
    a = 2.5
    for i in range(nbCircles):
        if  160 < features[i,0] <240 and 160 < features[i,1] <200 and 180 < features[i,2] <205:
            selectedCircles[i]=1
    circles = circles[selectedCircles]

    #Show final result
    showCircles(img, circles)    
    return circles
        
    
def getAverageColorInCircle(img, cx, cy, radius):
    '''
    Get the average color of img inside the circle located at (cx,cy) with radius.
    '''
    maxy,maxx,channels = img.shape
    nbVoxels = 0
    C = np.zeros( (3) )
    for dx in range( max(-cx,-radius),min(maxx-cx,+radius) ):
        for dy in range( max(-cy,-radius),min(maxy-cy,+radius) ):
            if (math.sqrt((dx)**2 + (dy)**2) < radius**2):
                nbVoxels += 1
                C = C + img[cy+dy, cx+dx, :]
    C = C/nbVoxels;
    return C
    
    
    
def showCircles(img, circles, text=None):
    '''
    Show circles on an image.
    @param img:     numpy array
    @param circles: numpy array 
                    shape = (nb_circles, 3)
                    contains for each circle: center_x, center_y, radius
    @param text:    optional parameter, list of strings to be plotted in the circles
    '''
    #make a copy of img
    img = np.copy(img)
    #draw the circles
    nbCircles = circles.shape[0]
    for i in range(nbCircles):
        cv2.circle(img, (int(circles[i,0]), int(circles[i,1])), int(circles[i,2]), cv2.cv.CV_RGB(255, 0, 0), 2, 8, 0 )
    #draw text
    if text!=None:
        for i in range(nbCircles):
            cv2.putText(img, text[i], (int(circles[i,0]), int(circles[i,1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv2.cv.CV_RGB(0, 0,255) )
    #show the result
    cv2.imshow('img',img)
    cv2.waitKey(0)    


        
if __name__ == '__main__':
    #read an image
    img = cv2.imread('normal.jpg')
    
    #print the dimension of the image
    print img.shape
    
    #show the image
    cv2.imshow('img',img)
    cv2.waitKey(0)
    
    #do detection
    circles = detect(img)
    
    #print result
    print "We counted "+str(circles.shape[0])+ " cells."
    








