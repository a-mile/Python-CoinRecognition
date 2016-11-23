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
    T = np.float32([[1, 0, w], [0, 1, h], [0, 0, 1]])
    R = cv2.getRotationMatrix2D((center[0]+w,center[1]+h), angle, 1.0)
    R = np.append(R, np.array([[0,0,1]]), axis=0)
    M = R.dot(T)
    M = M[:-1,:]
    rotated_image = cv2.warpAffine(image, M, (4 * w, 4 * h))
    return rotated_image

def crop_image(image, minAreaRect):
    (h, w) = image.shape[:2]
    rotate = rotateImage(image, minAreaRect[2],minAreaRect[0])
    return rotate[h + minAreaRect[0][1]- minAreaRect[1][1]//2 : h + minAreaRect[0][1] + minAreaRect[1][1]//2, w + minAreaRect[0][0] -  minAreaRect[1][0]//2 : w + minAreaRect[0][0] + minAreaRect[1][0]//2]

def avgs(color):
    color = color.astype(float)
    color[:,:,0][color[:,:,0] == 0] = np.nan
    color[:, :, 1][color[:, :, 1] == 0] = np.nan
    color[:, :, 2][color[:, :, 2] == 0] = np.nan

    return [math.ceil(np.nanmean(color[:,:,0])), math.ceil(np.nanmean(color[:,:,1])), math.ceil(np.nanmean(color[:,:,2]))]

def stds(color):
    color = color.astype(float)
    color[:, :, 0][color[:, :, 0] == 0] = np.nan
    color[:, :, 1][color[:, :, 1] == 0] = np.nan
    color[:, :, 2][color[:, :, 2] == 0] = np.nan

    return [math.ceil(np.nanstd(color[:, :, 0])), math.ceil(np.nanstd(color[:, :, 1])),
            math.ceil(np.nanstd(color[:, :, 2]))]

cv2.namedWindow('image')

dir = os.path.dirname(__file__)
imageNumber = 1
imageCount = 12

showImage = True

colorSpacesCount = 5
classNames = ['Dwadziescia zlotych','Dziesiec zlotych','Piec zlotych','Dwa zlote','Jeden zloty','Dwadziescia groszy','Dziesiec groszy']

data = np.load('data.npy')
avgsMin = data[0]
avgsMax = data[1]
stdsMin = data[2]
stdsMax = data[3]

while(1):
    filename = dir + '/samples/sample' + str(imageNumber) + '.jpg'

    image = cv2.imread(filename)
    image = cv2.resize(image,(1280,720))
    ims = image.copy()

    blur = cv2.GaussianBlur(image, (25, 25), 0)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    yuv = cv2.cvtColor(blur, cv2.COLOR_BGR2YUV)
    ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)
    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)

    thresh1 = cv2.Canny(blur[:, :, 0], 0,34)
    thresh2 = cv2.Canny(blur[:, :, 1], 0,25)
    thresh3 = cv2.Canny(blur[:, :, 2], 0,24)
    thresh4 = cv2.Canny(hsv[:, :, 1], 0, 46)
    thresh5 = cv2.Canny(hsv[:, :, 2], 0, 32)
    thresh6 = cv2.Canny(yuv[:, :, 0], 0, 22)
    thresh7 = cv2.Canny(yuv[:, :, 1], 0, 12)
    thresh8 = cv2.Canny(yuv[:, :, 2], 0, 27)
    thresh9 = cv2.Canny(ycrcb[:,:,0],0,32)
    thresh10 = cv2.Canny(ycrcb[:, :, 1], 0, 13)
    thresh11 = cv2.Canny(ycrcb[:, :, 2], 0, 14)
    thresh12 = cv2.Canny(gray,0,32)
    thresh13 = cv2.Canny(hls[:, :, 1], 0, 29)

    thresh = thresh1 | thresh2 | thresh3 | thresh4 | \
             thresh5 | thresh6 | thresh7 | thresh8 | \
             thresh9 | thresh10 | thresh11 | thresh12 | thresh13

    kernel = np.ones((10,10), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    cont_img = thresh.copy()
    _, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    rects = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])

        if area < 1000:
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
            if (not np.array_equal(tmp_box, boxes[i])) and is_in_rectangle(boxes[i], tmp_box):
                flag = True
                break
        if not flag:
            cropped_thresh = crop_image(thresh, rects[i])
            cropped = crop_image(image, rects[i])

            im_floodfill1 = cropped_thresh.copy()
            im_floodfill2 = cropped_thresh.copy()
            im_floodfill3 = cropped_thresh.copy()
            im_floodfill4 = cropped_thresh.copy()

            h, w = cropped_thresh.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)

            cv2.floodFill(im_floodfill1, mask, (0, 0), 255);
            cv2.floodFill(im_floodfill2, mask, (0, cropped_thresh.shape[0]-1), 255);
            cv2.floodFill(im_floodfill3, mask, (cropped_thresh.shape[1]-1, 0), 255);
            cv2.floodFill(im_floodfill4, mask, (cropped_thresh.shape[1]-1, cropped_thresh.shape[0]-1), 255);

            im_floodfill = im_floodfill1 | im_floodfill2| im_floodfill3| im_floodfill4

            im_floodfill_inv = cv2.bitwise_not(im_floodfill)

            im_out = cropped_thresh | im_floodfill_inv

            for ic in range(0,cropped.shape[0]):
                for jc in range(0, cropped.shape[1]):
                    if im_out[ic][jc] == 0:
                        cropped[ic][jc] = (0,0,0)

            colors = []

            colors.append(cropped)
            colors.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV))
            colors.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2YCR_CB))
            colors.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV))
            colors.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2HLS))

            classLens = []
            for k in range(0, len(classNames)):
                classLen = 0
                isClass = True
                for l in range(0, colorSpacesCount):
                    colorAvg = avgs(colors[l])
                    colorStd = stds(colors[l])
                    for m in range(0, 3):
                        if avgsMax[k][l][m] < colorAvg[m]:
                            isClass = False
                            classLen = -1
                            break
                        if stdsMax[k][l][m] < colorStd[m]:
                            isClass = False
                            classLen = -1
                            break
                        if avgsMin[k][l][m] > colorAvg[m]:
                            isClass = False
                            classLen = -1
                            break
                        if stdsMin[k][l][m] > colorStd[m]:
                            isClass = False
                            classLen = -1
                            break
                        classLen = classLen + abs((avgsMax[k][l][m] - avgsMin[k][l][m])/2 - colorAvg[m]) \
                            + abs((stdsMax[k][l][m] - stdsMin[k][l][m])/2 - colorStd[m])
                    if classLen == -1:
                        break
                classLens.append(classLen)

            max = 0
            index = -1
            for cl in range(0, len(classLens)):
                if classLens[cl] > max:
                    max = classLens[cl]
                    index = cl

            if index != -1:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(ims, classNames[index], (int(rects[i][0][0]) - 50, int(rects[i][0][1])), font, 0.5,(255, 0, 0), 2, cv2.LINE_AA)
                cv2.drawContours(ims, [boxes[i]], 0, (0, 0, 255), 2)

    if showImage:
        cv2.imshow('image', ims)
    else:
        cv2.imshow('image', thresh)

    key = cv2.waitKey(1000) & 0xFF
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