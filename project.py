import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os
import time


def callback(input):
    pass

def x_intersect(rho, theta):
    # Schnittpunkt mit y = 0
    return rho / np.cos(theta)

def y_intersect(rho, theta):
    # Schnittpunkt mit x = 0
    return rho / np.sin(theta)

def cornerDetection():
    root = os.getcwd()
    imgPath = os.path.join(root, "images\\test2.jpeg")
    og_img = cv.imread(imgPath)

    img = cv.cvtColor(og_img,cv.COLOR_BGR2RGB)

    height,width,_ = img.shape
    scale = 1/5
    heightScale = int(height*scale)
    widthScale = int(width*scale)
    img = cv.resize(img,(widthScale,heightScale), interpolation=cv.INTER_LINEAR)
    og_img = cv.resize(img,(widthScale,heightScale), interpolation=cv.INTER_LINEAR)
    overlay = og_img.copy()
    name = "canny"
    cv.namedWindow(name)

    #cv.createTrackbar("minThresh",name,0,255,callback)
    #cv.createTrackbar("maxThresh",name,0,255,callback)

    minThresh = 150
    maxThresh = 250
    
    cannyEdge = cv.Canny(img, minThresh,maxThresh)
    lines = cv.HoughLines(cannyEdge, 1, np.pi/180,80)
    
    paperEdges = []
    col = [(0,0,0),(0,255,0),(255,0,0),(255,0,255)]
    horizontal = []
    vertical = []
    verticalAbsRho = []
    horizontalAbsRho = [] 
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            print("Rho: " + str(rho))
            theta = lines[i][0][1]
            print("Theta: " + str(theta))
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            #cv.line(overlay, pt1, pt2, col[i], 3, cv.LINE_AA)

            absRho = rho
            if theta > np.pi/2.0:
                absRho = abs(rho)
            #    lines[i][0][0] = rho
    
            if(((theta <= np.pi*0.25 and theta >=0)  or (theta > np.pi*1.75 )) or (theta > np.pi*0.75 and theta <= np.pi*1.25)):
                print("vertical")
                vertical.append(lines[i])
                verticalAbsRho.append(absRho)
                #cv.line(overlay, pt1, pt2, (255,0,0), 3, cv.LINE_AA)
            else:
                print("horizontal")
                horizontal.append(lines[i])
                horizontalAbsRho.append(absRho)
                #cv.line(overlay, pt1, pt2, (0,255,0), 3, cv.LINE_AA)
    print(len(vertical), len(horizontal))

    verticalRho = np.array(verticalAbsRho, dtype=np.float32).reshape(-1, 1)
    horizontalRho = np.array(horizontalAbsRho, dtype=np.float32).reshape(-1, 1)

    print(verticalRho)
    vertLabels = KMeans(n_clusters=2, n_init=1).fit_predict(verticalRho)
    print(vertLabels)
    horLabels = KMeans(n_clusters=2, n_init=1).fit_predict(horizontalRho)
    print(horLabels)

    categorizedLines = [[],[],[],[]]

    for i,l in enumerate(vertLabels):
        if(l == 0):
            categorizedLines[0].append(vertical[i])
        else:
            categorizedLines[1].append(vertical[i])

    for i,l in enumerate(horLabels):
        if(l == 0):
            categorizedLines[2].append(horizontal[i])
        else:
            categorizedLines[3].append(horizontal[i])

    #bestLines = [leftVert, rightVert, topHorz, bottomHorz]
    bestLines = [None,None,None,None]
    for i,cat in enumerate(categorizedLines):
        #calc mean rho
        rhoSum = 0.0
        for j in range(len(cat)):
            rhoSum += cat[j][0][0]
        meanrho = rhoSum/len(cat)

        #get best line
        delta = 10000
        bestIdx = -1
        for j in range(len(cat)):
            if(abs(meanrho - cat[j][0][0]) < delta):
                delta = abs(meanrho - cat[j][0][0])
                bestIdx = j

        bestLine = cat[bestIdx]

        rho = bestLine[0][0]
        theta = bestLine[0][1]
        #sort into bestLines[]
        if(i < 2):
            #vertical
            xi = x_intersect(rho, theta)

            if bestLines[0] is None:
                bestLines[0] = bestLine
            else:
                # Vergleiche x-Schnittpunkte
                rho0 = bestLines[0][0][0]
                theta0 = bestLines[0][0][1]
                xi0 = x_intersect(rho0, theta0)

                if xi < xi0:
                    bestLines[1] = bestLines[0]
                    bestLines[0] = bestLine
                else:
                    bestLines[1] = bestLine
        else:
            #horizontal
            yi = y_intersect(rho, theta)

            if bestLines[2] is None:
                bestLines[2] = bestLine
            else:
                rho2 = bestLines[2][0][0]
                theta2 = bestLines[2][0][1]
                yi0 = y_intersect(rho2, theta2)

                if yi < yi0:
                    bestLines[3] = bestLines[2]
                    bestLines[2] = bestLine
                else:
                    bestLines[3] = bestLine

    print(bestLines)
    for i in range(len(bestLines)):
        rho = bestLines[i][0][0]
        theta = bestLines[i][0][1]
        print(rho)
        print(theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        print(pt1)
        print(pt2)
        #cv.line(overlay, pt1, pt2, col[i], 3, cv.LINE_AA)
        h, w = overlay.shape[:2]
        ok, clipped_pt1, clipped_pt2 = cv.clipLine((0, 0, w, h), pt1, pt2)

        if ok:
            cv.line(overlay, clipped_pt1, clipped_pt2, col[i], 3, cv.LINE_AA)

   
    contours, hierarchy = cv.findContours(cannyEdge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    
    while True:
        if cv.waitKey(1) == ord("q"):
            break

        #cv.drawContours(overlay, contours, -1, (0,255,0), 2)

        cv.imshow(name, cannyEdge)
        cv.imshow("Overlay", overlay)

    cv.destroyAllWindows()

if __name__ == "__main__":
    #KMeans warm up
    dummy = np.array([[0],[1]], dtype=np.float32)
    KMeans(n_clusters=2, n_init=1).fit_predict(dummy)

    cornerDetection()