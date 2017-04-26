import os,sys
import cv2
import numpy as np
#cv2.imread()




def cropToLandmarks(dir,startTooth,endTooth,extraSpace):
    for j in range(1, len(dir) / 8 + 1):  # There are 8 files for each radiograph
        coords = []
        for i in range(startTooth, endTooth):  # Take all coords together and find a window
            file = open("projectdata/data/Landmarks/original/landmarks" + str(j) + "-" + str(i) + ".txt", 'r')
            newcoords = file.read()
            newcoords = newcoords.split("\n")
            newcoords.pop()
            y = len(newcoords)
            newcoordsnum = [int(float(x)) for x in newcoords]
            coords.extend(newcoordsnum)
        l = len(coords)
        indexx = [2 * x for x in range(len(coords) / 2)]
        xcoords = [coords[2 * x] for x in range(len(coords) / 2)]
        ycoords = [coords[2 * x + 1] for x in range(len(coords) / 2)]
        xcoords = np.array(xcoords)
        ycoords = np.array(ycoords)
        xtop = np.amin(xcoords)
        ytop = np.amin(ycoords)
        xbot = np.amax(xcoords)
        ybot = np.amax(ycoords)
        if j < 10:
            index = "0" + str(j)
        else:
            index = str(j)
        filename = index + ".tif"
        location = "projectdata/data/Radiographs/" + filename
        img = cv2.imread(location)
        crop_img = img[ytop-extraSpace:ybot+extraSpace, xtop-extraSpace:xbot+extraSpace]
        # cv2.rectangle(img, (xtop, ytop), (xbot, ybot), (0, 255, 0), 3)
        # for i in range(len(xcoords)-1):
        #    cv2.line(img, (xcoords[i],ycoords[i]), (xcoords[i+1],ycoords[i+1]),(0, 255, 0))
        # imS = cv2.resize(img,(600,400))
        #cv2.imshow("img", crop_img)
        #cv2.waitKey(0)
        cv2.imwrite("projectdata/data/Frontteeth/"+index + ".tif",crop_img)

dir = os.listdir("projectdata/data/Landmarks/original")
cropToLandmarks(dir, 1, 5, 10)