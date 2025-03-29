import cv2
import numpy as np

def findContoursAndFilter(image, threshold=[100, 100], showEdges=False, minArea=1000, filterCount=0, drawContours=False):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 1)
    edgeImage = cv2.Canny(blurredImage, threshold[0], threshold[1])
    
    kernel = np.ones((5, 5))
    dilatedImage = cv2.dilate(edgeImage, kernel, iterations=3)
    thresholdImage = cv2.erode(dilatedImage, kernel, iterations=2)
    
    if showEdges:
        cv2.imshow("Edge Detection", thresholdImage)
    
    contours, _ = cv2.findContours(thresholdImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea:
            perimeter = cv2.arcLength(contour, True)
            approxCorners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            boundingBox = cv2.boundingRect(approxCorners)
            
            if filterCount > 0:
                print(approxCorners)
                if len(approxCorners) == filterCount:
                    finalContours.append((len(approxCorners), area, approxCorners, boundingBox, contour))
            else:
                finalContours.append((len(approxCorners), area, approxCorners, boundingBox, contour))
    
    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
    
    if drawContours:
        for cont in finalContours:
            cv2.drawContours(image, cont[4], -1, (0, 0, 255), 3)
        cv2.imshow("image with detected contours", image)
    return image, finalContours


def reorderPoints(points):
    points = points.reshape((4, 2))
    orderedPoints = np.zeros_like(points)
    
    pointSum = points.sum(axis=1)
    orderedPoints[0] = points[np.argmin(pointSum)]
    orderedPoints[3] = points[np.argmax(pointSum)]
    
    pointDiff = np.diff(points, axis=1)
    orderedPoints[1] = points[np.argmin(pointDiff)]
    orderedPoints[2] = points[np.argmax(pointDiff)]
    
    return orderedPoints


def warpPerspectiveImage(image, points, width, height, padding=20):
    orderedPoints = reorderPoints(points)
    
    srcPoints = np.float32(orderedPoints)
    dstPoints = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    
    transformationMatrix = cv2.getPerspectiveTransform(srcPoints, dstPoints)
    warpedImage = cv2.warpPerspective(image, transformationMatrix, (width, height))
    
    return warpedImage[padding:warpedImage.shape[0] - padding, padding:warpedImage.shape[1] - padding]


def calculateDistance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

webcamFeed = False
if webcamFeed:
    # Initialize camera
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(10, 160)
    cap.set(3, 1920)
    cap.set(4, 1080)
else:
    img_path = "./samples/20250120_203144.jpg"
scale_factor = 3
width_paper = 210 * scale_factor
height_paper = 297 * scale_factor

while True:
    if webcamFeed:
        ret, image = cap.read()
    else:
        image=cv2.imread(img_path)
    resizedImage = cv2.resize(image, (0, 0), None, 0.5, 0.5)
    imageWithContours, contours = findContoursAndFilter(resizedImage, showEdges=False, minArea=50000, filterCount=4,threshold=[120, 120], drawContours=False)
    
    if contours:
        largestContour = contours[0][2]
        warpedImage = warpPerspectiveImage(resizedImage, largestContour, width_paper, height_paper)
        
        # cv2.imshow("Warped A4 Paper", warpedImage)
        
        imageWithObjects, objectContours = findContoursAndFilter(resizedImage, showEdges=False, minArea=5000, filterCount=4, threshold=[100, 100], drawContours=False)
        
        if objectContours:
            for obj in objectContours:
                cv2.polylines(imageWithObjects, [obj[2]], True, (0, 255, 0), 2)
                orderedPoints = reorderPoints(obj[2])
                # Calculate object width and height
                objectWidth = round(calculateDistance(orderedPoints[0]/scale_factor, orderedPoints[1]/scale_factor) / 10, 1)
                objectHeight = round(calculateDistance(orderedPoints[0]/scale_factor, orderedPoints[2]/scale_factor) / 10, 1)
                
                # Draw dimension arrows and labels
                cv2.arrowedLine(imageWithObjects, (orderedPoints[0][0], orderedPoints[1][0]), (orderedPoints[1][0], orderedPoints[1][1]), (255, 0, 255), 3, 8, 9, 0.05)
                cv2.arrowedLine(imageWithObjects, (orderedPoints[0][0], orderedPoints[1][0]), (orderedPoints[2][0], orderedPoints[2][1]), (255, 0, 255), 3, 8, 9, 0.05)
                
                x, y, _, _ = obj[3]
                cv2.putText(imageWithObjects, "{}cm".format(objectWidth), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                cv2.putText(imageWithObjects, "{}cm".format(objectHeight), (x - 70, y + obj[3][3] // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

        cv2.imshow("Detected Objects with Dimensions", imageWithObjects)

    # # Resize for display
    # resizedImage = cv2.resize(image, (0, 0), None, 0.5, 0.5)
    
    # # Show original image
    # cv2.imshow("Original Image", resizedImage)
    
    # Check for 'q' key press to exit
    if webcamFeed and cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(0):
        break

# Release camera and close all windows
if webcamFeed:
    cap.release()
    cv2.destroyAllWindows()
