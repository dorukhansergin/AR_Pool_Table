import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from src.arpt_utils import ARPoolTable


images_dir = os.path.join("images", "corner_testing")
image_paths = [os.path.join(images_dir,name) for name in os.listdir(images_dir)]
top_keys = ['org', 'hsv', 'lab']
img_keys = [name.replace(".jpg","") for name in os.listdir(images_dir)]
imgs = {key: dict.fromkeys(img_keys) for key in top_keys}

for i, path in enumerate(image_paths):
    imgs['org'][img_keys[i]] = cv2.imread(path)

img_id = 5
img = imgs['org'][img_keys[img_id]]

arpt = ARPoolTable([0,110],[130,150],10,5)
arpt.table_detector(img)
arpt.plot_corners(img)
center = arpt.track_ball(img, [[0, 255], [160, 255]])
print(center)
