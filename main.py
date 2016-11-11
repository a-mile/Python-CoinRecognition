import numpy as np
import cv2
import os

dir = os.path.dirname(__file__)
imageNumber = 1
imageCount = 11

while(1):
    filename = dir + '/samples/sample'+str(imageNumber)+'.jpg'

    image = cv2.imread(filename,0)

    blur = cv2.GaussianBlur(image, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 1)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
    cont_img = closing.copy()
    _, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue
        cv2.drawContours(image, [cnt], 0, (0, 255, 0), 3)

    cv2.imshow('image',closing)

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