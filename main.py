import numpy as np
import cv2
import os

def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('threshold1','image',0,255,nothing)
cv2.createTrackbar('threshold2','image',0,255,nothing)

dir = os.path.dirname(__file__)
imageNumber = 1
imageCount = 11

while(1):
    filename = dir + '/samples/sample'+str(imageNumber)+'.jpg'

    image = cv2.imread(filename)

    blur = cv2.GaussianBlur(image, (15, 15), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv,(cv2.getTrackbarPos('threshold1','image'),0,0),(cv2.getTrackbarPos('threshold2','image'),255,255))

    cont_img = thresh.copy()
    _, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue
        cv2.drawContours(image, [cnt], 0, (0, 255, 0), 3)

    cv2.imshow('image',image)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == 3:
        imageNumber = imageNumber + 1
        if imageNumber > imageCount:
            imageNumber = imageCount
    if key == 2:
        imageNumber = imageNumber - 1
        if imageNumber < 1:
            imageNumber = 1

cv2.destroyAllWindows()