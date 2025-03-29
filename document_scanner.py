import cv2
import numpy as np

def getContours(img, cThr=[100,100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    thr = valTrackbars()
    imgCanny = cv2.Canny(imgBlur, thres[0],thres[1])
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel, iterations=2)
    imgThre = cv2.erode(imgDial,kernel, iterations=1)
    imgContours = img.copy()
    imgBigContours = img.copy()
   
    contours, hiearchy = cv2.findContours(imgThre,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
    biggest,maxArea = biggestContour(contours)
    if biggest.size!=0:
        biggest=reorder(biggest)
        cv2.drawContours(imgBigContours, biggest, 2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        imgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))
       
       
       
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea()
        if area>5000:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            if area>max_area and len(approx)==4:
                biggest=approx
                max_area=area
    return biggest, max_area
   
           
    finalContours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            bbox = cv2.boundingRect(approx)
            if filter>0:
                if len(approx)==filter:
                    finalContours.append(len(approx), area, approx,bbox, i)
            else:
                finalContours.append(len(approx), area, approx, bbox, i)
    finalContours = sorted(finalContours,key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img,con[4],-1,(0,0,255),3)
    return img, finalContours
# def reorder(myPoints):
#     print(myPoints.shape)
#     myPointsNew = np.zeros_like(myPoints)
#     myPoints = myPoints.reshape((4,2))
#     add = myPoints.sum(1)
#     myPointsNew[0] = myPoints[np.argmin(add)]
#     myPointsNew[3] = myPoints[np.argmax(add)]
#     diff = np.diff(myPoints,axis=1)
#     myPointsNew[1] = myPoints[np.argmin(diff)]
#     myPointsNew[2] = myPoints[np.argmax(diff)]
#     return myPointsNew
# def warpImg(img, points, w,h, pad=20):
#     print(points)
#     points = reorder(points)
#     pts1 = np.float32(points)
#     pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
#     matrix = cv2.getPerspectiveTransform(pts1,pts2)
#     imgWarp = cv2.warpPerspective(img,matrix,(w,h))
#     imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]
#     return imgWarp
#
# def findDis(pts1,pts2):
#     return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5

webcamFeed = True
pathImg = "1.jpg"
cap = cv2.VideoCapture(0)
cap.set(10,160)
heightImg = 640
widthImg = 480
# cap.set(3,1920)
# cap.set(4,1080)
# scale=3
# wP=210*scale
# hP = 297*scale
initializeTrackbars()
while True:
    imgBlank = np.zeros((heightImg,widthImg,3), np.uint8)
    if webcamFeed:
        success, img = cap.read()
    else:
        img=cv2.imread(pathImg)
    img = cv2.resize(img, (widthImg, heightImg))

    imgContours, conts = getContours(img,showCanny=True, minArea=50000, filter=4, draw=True)
    if len(conts)!=0:
        biggest = conts[0][2]
        print(biggest)
        imgWarp = warpImg(img,biggest,wP,hP)
        cv2.imshow("A4", imgWarp)
        imgContours2, conts2 = getContours(img, showCanny=True, minArea=2000, filter=4, cThr=[50,50], draw=True)
        if len(conts2)!=0:
            for obj in conts2:
                cv2.polylines(imgContours2,[obj[2]],True,(0,255,0),2)
                nPoints = reorder(obj[2])
                mW = round(findDis(nPoints[0][0]//scale, nPoints[1][0]//scale)/10,1)
                nH = round(findDis(nPoints[0][0]//scale, nPoints[2][0]//scale)/10,1)
                cv2.arrowedLine(imgContours2,(nPoints[0][0][0], nPoints[0][0][1]),(nPoints[1][0][0], nPoints[1][0][1],
                                                                          (255,0,255),3,8,9,0.05))
                cv2.arrowedLine(imgContours2,(nPoints[0][0][0], nPoints[0][0][1]),(nPoints[2][0][0], nPoints[2][0][1],
                                                                          (255,0,255),3,8,9,0.05))
                x,y,w,h = obj[3]
                cv2.putText(imgContours2,"{}cm".format(mW), (x+30,y-10),cv2.FONT_HERSHEY_COMPLEX_SMA,(255,0,255),2)
                cv2.putText(imgContours2,"{}cm".format(nH), (x-70,y+h//2),cv2.FONT_HERSHEY_COMPLEX_SMA,(255,0,255),2)

        cv2.imshow("A4", imgContours2)

    img = cv2.resize(img,(0,0),None,0.5,0.5)

    cv2.imshow("original",img)
    cv2.waitKey(1)
