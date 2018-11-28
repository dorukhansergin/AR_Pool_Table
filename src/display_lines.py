import os
import functools
import matplotlib.pyplot as plt
import cv2
import numpy as np
from src.arpt_utils import ARPoolTable
from src.arpt_aruco import ARPT

arpt = ARPT([75, 115],[140, 170])

image_dir = '../images/aruco/'

img = cv2.imread(image_dir+'ex_cue30.jpg')
print(img.shape)

# resize here
# img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
'''
# detect corners and warp
corners = arpt.detectCorners(img)
print(corners)
warped = arpt.warpToRect(img,corners)

# draw a line on the warped image
cv2.line(warped,(0,0),(150,150),(255,255,255),2)
'''
# get binary image and detect the cue ball
center, radius = arpt.tableSegmentation(img)

# detect the cue around the ball

# Step 1 draw a rectangular region around the center of the ball
scaling_factor = 3
cv2.rectangle(img,(center[0]-scaling_factor*radius,center[1]+scaling_factor*radius),
              (center[0] + scaling_factor * radius, center[1] - scaling_factor * radius),(255,255,255),2)

# threshold the portion of image corresponding to the cue, Y X order in array of image
mini = img[center[1]-scaling_factor*radius:center[1]+scaling_factor*radius,
       center[0]-scaling_factor*radius:center[0]+scaling_factor*radius,:]

lab = cv2.cvtColor(mini, cv2.COLOR_BGR2Lab)
a_img = arpt.ab_threshold(lab, "a")
b_img = arpt.ab_threshold(lab, "b")
ab_img = np.logical_and(a_img, b_img).astype("uint8")
# close the img
inv_img = ~ab_img
# perform morphological operations
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(inv_img, cv2.MORPH_OPEN, kernel, iterations=3)
# draw the contours on the opened image
pad = 1
constant= cv2.copyMakeBorder(opening,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=[0,0])
_, contours, _ = cv2.findContours(~constant, cv2.RETR_TREE, cv2.cv2.CHAIN_APPROX_SIMPLE)


myfunc = functools.partial(cv2.arcLength, closed=True)

sort_contours = sorted(contours,key=myfunc)
# eliminates the contour corresponding to the border
sort_contours.pop()
# eliminate contour corresponding to ball
for contour in sort_contours:
    is_ball=cv2.pointPolygonTest(contour, (mini.shape[0]//2,mini.shape[1]//2), False)
    print(is_ball)
    if is_ball != 1:
        stick_contour = contour
        break
# try catch this in implementation
cv2.drawContours(mini,[stick_contour],-1,(0,0,255),3)

# approximate a polygon around the stick contour
rect = cv2.minAreaRect(stick_contour)
vec = center - np.array(rect[0])
angle = np.arctan2(vec[1],vec[0]) * 180 / np.pi
print(rect[2] % 90, angle % 90)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(mini,[box],0,(0,255,255),2)
# plt.matshow(opening)
# plt.show()
# decide if cue coincides with ball


# draw predicted line

# measure error


# draw the min-enclosing cirle on the warped img
cv2.circle(img,center,radius,(0,255,255))

cv2.imshow('warp',img)
cv2.imshow('mini',mini)
cv2.waitKey(0)
cv2.destroyAllWindows()