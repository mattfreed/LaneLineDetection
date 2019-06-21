import numpy as np
import cv2

image = cv2.imread('Picture/test_image.jpg')
lane_image = np.copy(image)
gray = cv2.cvtColor(lane_image)
cv2.imshow('result', image)
cv2.waitKey(0)

