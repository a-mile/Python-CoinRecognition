import numpy as np
import cv2
import os

cv2.namedWindow('image')

dir = os.path.dirname(__file__)
imageNumber = 1
imageCount = 11
showImage = True

while(1):
    filename = dir + '/samples/sample'+str(imageNumber)+'.jpg'

    image = cv2.imread(filename)

    blur = cv2.GaussianBlur(image, (35, 35), 0)

    gray = hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 1)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

    cont_img = thresh.copy()
    _, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    if showImage:
        cv2.imshow('image',image)
    else:
        cv2.imshow('image', thresh)

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
    if key == 1:
        showImage = not showImage

cv2.destroyAllWindows()