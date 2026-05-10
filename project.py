import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os
import time



def callback(input):
    pass

def cannyEdge():
    root = os.getcwd()
    imgPath = os.path.join(root, "images\\test1.jpeg")
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
    col = [(0,0,255),(0,255,0),(255,0,0), (255,0,255),(255,0,255),(0,255,255),(255,255,255),(0,0,0),(100,100,100)]
    horizontal = []
    vertical = []
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            print(col[i])
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

            if theta > np.pi/2.0:
                rho = abs(rho)
                lines[i][0][0] = rho
    
            if(((theta <= np.pi*0.25 and theta >=0)  or (theta > np.pi*1.75 )) or (theta > np.pi*0.75 and theta <= np.pi*1.25)):
                print("vertical")
                vertical.append(lines[i])
                #cv.line(overlay, pt1, pt2, (255,0,0), 3, cv.LINE_AA)
            else:
                print("horizontal")
                horizontal.append(lines[i])
                #cv.line(overlay, pt1, pt2, (0,255,0), 3, cv.LINE_AA)
    print(len(vertical), len(horizontal))

    print("start kmeans")
    verticalRho = np.array([vert[0][0] for vert in vertical], dtype=np.float32).reshape(-1, 1)
    horizontalRho = np.array([hor[0][0] for hor in horizontal], dtype=np.float32).reshape(-1, 1)
    

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
        #sort into bestLines[]
        if(i < 2):
            #vertical
            if bestLines[0] is None:
                bestLines[0] = cat[bestIdx]
            else:
                if(bestLines[0][0][0] > cat[bestIdx][0][0]):
                    bestLines[1]= bestLines[0]
                    bestLines[0] = cat[bestIdx]
                else:
                    bestLines[1] = cat[bestIdx]
        else:
            #horizontal
            if bestLines[2] is None:
                bestLines[2] = cat[bestIdx]
            else:
                if(bestLines[2][0][0] > cat[bestIdx][0][0]):
                    bestLines[3]= bestLines[2]
                    bestLines[2] = cat[bestIdx]
                else:
                    bestLines[3] = cat[bestIdx]

    print(bestLines)
    for i in range(len(bestLines)):
        rho = bestLines[i][0][0]
        theta = bestLines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(overlay, pt1, pt2, col[i], 3, cv.LINE_AA)

   
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

    cannyEdge()