import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from src.arpt_utils import ARPoolTable


images_dir = os.path.join("images", "cue_stick")
image_paths = [os.path.join(images_dir,name) for name in os.listdir(images_dir)]
top_keys = ['org', 'hsv', 'lab']
img_keys = [name.replace(".jpg","") for name in os.listdir(images_dir)]
imgs = {key: dict.fromkeys(img_keys) for key in top_keys}

for i, path in enumerate(image_paths):
    imgs['org'][img_keys[i]] = cv2.imread(path)

img_id = 2
img = imgs['org'][img_keys[img_id]]

arpt = ARPoolTable([0, 110],[130, 150], 5, 5)

# plt.imshow(img)
# plt.show()
# arpt.steps_and_plots(img)
# arpt.table_detector(img)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
# 1: 110, 130 - 2: 120 130
a_l = lab[:,:,1] > 120
a_2 = lab[:,:,1] < 135
b_l = lab[:,:,2] > 135
b_2 = lab[:,:,2] < 150
a_img = np.logical_and(a_l, a_2)
b_img = np.logical_and(b_l, b_2)
ab_img = np.logical_and(a_img, b_img)
plt.matshow(a_img)
plt.show()
plt.matshow(b_img)
plt.show()
plt.matshow(ab_img)
plt.show()

