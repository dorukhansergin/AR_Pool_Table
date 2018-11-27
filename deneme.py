from __future__ import print_function

import os
import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt

from src.arpt_aruco import ARPT
video_dir = os.path.join("images", "aruco", "vid2.mp4")

arpt = ARPT()
pt = arpt.runVideo(video_dir)


# id_order = np.array([6,1,2,3])
# shuffled = np.array([2,3,1,6])
# shuffled_corners = np.array([1,2,3,4])
# print([shuffled_corners[np.where(shuffled==id)] for id in id_order])






# for i, path in enumerate(image_paths):
#     imgs['org'][img_keys[i]] = cv2.imread(path)
#
# img_id = 2
# img = imgs['org'][img_keys[img_id]]
#
# arpt = ARPoolTable([0, 110],[130, 150], 5, 5)
#
# plt.imshow(img)
# plt.show()


# arpt.table_detector(img)
# plt.imshow(img)
# for corner in corners:
#     plt.scatter(corner[0], corner[1])
# plt.show()

# for pair in pairs:
#     x1, x2, y1, y2 = poly[pair[0],0], poly[pair[1],0], poly[pair[0],1], poly[pair[1],1]
#     plt.plot([x1,x2], [y1,y2])
# plt.show()
# hull = np.squeeze(hull, 1)
# poly = np.squeeze(poly, 1)

# cv2.drawContours(img, hull, 1, (0, 0, 255), 1, 8)
# plt.imshow(img); plt.show()
# fig, ax = plt.subplots(2,2)
# ax[0,0].imshow(img)
# ax[0,0].scatter(hull[:,0], hull[:,1], color="r")
# ax[0,1].imshow(img)
# ax[0,1].scatter(poly[:,0], poly[:,1], color="r")
# ax[1,0].matshow(opening)
# ax[1,1].matshow(closing)
#
# fig.show()
# plt.waitforbuttonpress()

# ab_img, opening, cnt, hull = arpt.table_detector(img)


# arpt.plot_corners(img)
# center = arpt.track_ball(img, [[0, 255], [160, 255]])
# plt.imshow(img)
# plt.scatter(center[0], center[1], color="r")
# plt.show()
