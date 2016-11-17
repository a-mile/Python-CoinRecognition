import numpy as np
import cv2
import os
import math
import matplotlib.path as mplPath
from scipy.spatial import distance as dist

def is_in_rectangle(first, rectangle):
    rectangle = order_points(rectangle)
    rectanglePath = mplPath.Path(rectangle)
    result = rectanglePath.contains_points(first)
    count = 0
    for bl in result:
        if bl:
            count += 1
    if count > 2:
        return True
    return False

def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")

def rotateImage(image, angle, center):
    (h, w) = image.shape[:2]
    T = np.float32([[1, 0, w/2], [0, 1, h/2], [0, 0, 1]])
    # rotated_image = cv2.warpAffine(image, M, (2 * w, 2 * h))
    R = cv2.getRotationMatrix2D((center[0]+w/2,center[1]+h/2), angle, 1.0)
    R = np.append(R, np.array([[0,0,1]]), axis=0)
    M = R.dot(T)
    M = M[:-1,:]
    rotated_image = cv2.warpAffine(image, M, (2 * w, 2 * h))
    return rotated_image

def crop_image(image, minAreaRect):
    (h, w) = image.shape[:2]
    rotate = rotateImage(image, minAreaRect[2],minAreaRect[0])
    return rotate[h/2 + minAreaRect[0][1]- minAreaRect[1][1]/2 : h/2 + minAreaRect[0][1] + minAreaRect[1][1]/2, w/2 + minAreaRect[0][0] -  minAreaRect[1][0]/2 : w/2 + minAreaRect[0][0] + minAreaRect[1][0]/2]

def avgs(color):
    return [np.average(color[:,:,0]),np.average(color[:,:,1]),np.average(color[:,:,2])]

def stds(color):
    return [np.std(color[:,:,0]),np.std(color[:,:,1]),np.std(color[:,:,2])]

cv2.namedWindow('image')

dir = os.path.dirname(__file__)
imageNumber = 1
imageCount = 11

colorSpacesCount = 5
classNames = ['Dwadziescia zlotych','Dziesiec zlotych','Piec zlotych','Dwa zlote','Jeden zloty','Dwadziescia groszy','Dziesiec groszy']

data = np.load('data.npy')
avgsMin = data[0]
avgsMax = data[1]
stdsMin = data[2]
stdsMax = data[3]


while(1):
    filename = dir + '/samples/sample'+str(imageNumber)+'.jpg'

    image = cv2.imread(filename)

    blur = cv2.GaussianBlur(image, (35, 35), 0)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 1)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

    cont_img = thresh.copy()
    _, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    rects = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])

        if area < 2000:
            continue
        if hierarchy[0][i][3] != -1:
            continue

        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxes.append(box)
        rects.append(rect)

    for i in range(len(boxes)):
        flag = False
        for tmp_box in boxes:
            flag = False
            if (not np.array_equal(tmp_box,boxes[i])) and is_in_rectangle(boxes[i],tmp_box):
                flag = True
                break
        if not flag:
            cropped = crop_image(image, rects[i])
            file = dir + '/cropped/crop' + str(imageNumber) + str(i) + '.jpg'
            cv2.imwrite(file, cropped)
            cropped = cv2.imread(file)

            colors = []

            colors.append(cropped)
            colors.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV))
            colors.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2YCR_CB))
            colors.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV))
            colors.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2HLS))

            for k in range(0,len(classNames)):
                isClass = True
                for l in range(0, colorSpacesCount):
                    colorAvg = avgs(colors[l])
                    colorStd = stds(colors[l])
                    for m in range(0, 3):
                        if math.ceil(avgsMax[k][l][m]) < colorAvg[m]:
                            isClass = False
                            break
                        if math.ceil(stdsMax[k][l][m]) < colorStd[m]:
                            isClass = False
                            break
                        if math.floor(avgsMin[k][l][m]) > colorAvg[m]:
                            isClass = False
                            break
                        if  math.floor(stdsMin[k][l][m]) > colorStd[m]:
                            isClass = False
                            break
                if isClass:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, classNames[k], (int(rects[i][0][0])-50,int(rects[i][0][1])), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.drawContours(image, [boxes[i]], 0, (0, 0, 255), 2)

    cv2.imshow('image', image)

    key = cv2.waitKey(100) & 0xFF
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