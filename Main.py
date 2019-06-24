import numpy as np
import cv2
import matplotlib.pyplot as plt

def cannyDisplay (image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def regionOfInterest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    maskedImage = cv2.bitwise_and(image, mask)
    return maskedImage

def display_Lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def averageSlopedIntercept(image, liens):
    leftFit = []
    rightFit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope <0:
            leftFit.append((slope,intercept))
        else:
            rightFit.append((slope,intercept))
    leftFitAverage = np.average(leftFit,axis = 0)
    rightFitAverage = np.average(rightFit, axis = 0)
    leftLine = makeCoordinates(image, leftFitAverage)
    rightLine = makeCoordinates(image, rightFitAverage)
    return np.array([leftLine, rightLine])

def makeCoordinates(image, lineParameters):
    slope, intercept = lineParameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2,y2])



# image = cv2.imread('Picture/test_image.jpg')
# lane_image = np.copy(image)
# cannyImage = cannyDisplay(lane_image)
# croppedImage = regionOfInterest(cannyImage)
# lines = cv2.HoughLinesP(croppedImage, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# avgLines = averageSlopedIntercept(lane_image,lines)
# line_image = display_Lines(lane_image, avgLines)
# comboImage = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#
# cv2.imshow('result1', image)
# cv2.imshow('result2', croppedImage)
# cv2.imshow('result3', line_image)
# cv2.imshow('result4', comboImage)
# # cv2.waitKey(0)

video = cv2.VideoCapture("Video/test2.mp4")
while(video.isOpened()):
    _, frame = video.read()
    cannyImage = cannyDisplay(frame)
    croppedImage = regionOfInterest(cannyImage)
    lines = cv2.HoughLinesP(croppedImage, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    avgLines = averageSlopedIntercept(frame, lines)
    line_image = display_Lines(frame, avgLines)
    comboImage = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('video', comboImage)
    if cv2.waitKey(1) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()