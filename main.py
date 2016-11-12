import numpy as np
import cv2
import os

def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('C','image',0,10,nothing)
cv2.createTrackbar('Opening iterations','image',0,10,nothing)

dir = os.path.dirname(__file__)
imageNumber = 1
imageCount = 11
showImage = True

while(1):
    filename = dir + '/samples/sample'+str(imageNumber)+'.jpg'

    image = cv2.imread(filename)

    blur = cv2.GaussianBlur(image, (15, 15), 0)

    gray = hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, (cv2.getTrackbarPos('C','image')))
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=(cv2.getTrackbarPos('Opening iterations','image')))

    cont_img = thresh.copy()
    _, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue
        hull = cv2.convexHull(cnt)
        hullf = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hullf)

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.circle(image, far, 5, [0, 0, 255], -1)

        cv2.drawContours(image, [hull], -1, (0, 255, 0), 2)
        #cv2.drawContours(image, [cnt], 0, (255, 255, 0), 3)

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