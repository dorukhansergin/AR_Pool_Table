import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


class ARPoolTable:
    def __init__(self, a_thresh, b_thresh, morph_iter, kernel_size):
        # TODO: docstring
        self.a_thresh = a_thresh
        self.b_thresh = b_thresh
        self.morph_iter = morph_iter
        self.kernel_size = kernel_size
        self.maxWidth = 526
        self.maxHeight = 306
        self.corners = None
        self.warped = None

    def table_detector(self, img):
        # TODO: docstring
        # Convert Color to Lab space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        # TODO: Convert these to cv2.threshold, possibly faster
        # Color threshold on a and b colorspaces, then get an elementwise AND of both
        a_img = self.ab_threshold(lab, "a")
        b_img = self.ab_threshold(lab, "b")
        ab_img = np.logical_and(a_img, b_img).astype("uint8")

        # Apply opening and closing operations one after another
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        opening = cv2.morphologyEx(ab_img, cv2.MORPH_OPEN, kernel, iterations=self.morph_iter)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iter*3)

        # Find the convex hull of the contour and then approximate a polygon around it
        im2, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 1 # make sure we have a single component
        hull = cv2.convexHull(contours[0])
        poly = cv2.approxPolyDP(hull, 0.01*cv2.arcLength(hull,True), True)
        assert poly.shape[0] == 8 # make sure we get 4 corners

        # poly comes in shape [num_pts, 1, 2] so we get rid of index 1
        poly = np.squeeze(poly, 1)

        # get corners with this terrible procedure
        self.corners = self.get_corners_from_poly(poly)
        self.warped = self.warp_to_rect(img)

    def warp_to_rect(self, img):
        # code taken from https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/

        dst = np.array([
            [0, 0],
            [self.maxWidth - 1, 0],
            [0, self.maxHeight - 1],
            [self.maxWidth - 1, self.maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(self.corners, dst)
        warp = cv2.warpPerspective(img, M, (self.maxWidth, self.maxHeight))
        return warp

    def ab_threshold(self, lab, color_space):
        if color_space == "a":
            axis = 1
            thresh = self.a_thresh
        elif color_space == "b":
            axis = 2
            thresh = self.b_thresh
        else:
            raise ValueError("Invalid colorspace argument")
        return np.logical_and(lab[:, :, axis] > thresh[0], lab[:, :, axis] < thresh[1])

    @staticmethod
    def color_mask(lab, thresh, color_space):
        if color_space == "a":
            axis = 1
        elif color_space == "b":
            axis = 2
        else:
            raise ValueError("Invalid colorspace argument")
        return np.logical_and(lab[:, :, axis] > thresh[0], lab[:, :, axis] < thresh[1])

    def get_corners_from_poly(self, poly):
        corners = []
        num_points = poly.shape[0]
        first_pt_indices = list(range(num_points))
        second_pt_indices = list(range(1,num_points)) + [0]
        pt_pair_indices_list = list(zip(first_pt_indices, second_pt_indices))
        active_pairs = [True] * len(pt_pair_indices_list)
        max_pair_idx_list = []

        for _ in range(4):
            distances = np.array([euclidean(poly[first_pt_idx], poly[second_pt_idx])
                                  for first_pt_idx, second_pt_idx in pt_pair_indices_list])
            distances = [dist if active_pairs[i] else 0 for (i, dist) in enumerate(distances)]
            max_pair_idx = np.argmax(distances)
            max_pair_idx_list.append(max_pair_idx)
            if max_pair_idx == 0:
                deactive_idx = [-1,0,1]
            elif max_pair_idx == len(active_pairs)-1:
                deactive_idx = [len(active_pairs)-2, len(active_pairs)-1, 0]
            else:
                deactive_idx = [max_pair_idx-1, max_pair_idx, max_pair_idx+1]

            for idx in deactive_idx:
                active_pairs[idx] = False
        max_pairs = [pt_pair_indices_list[i] for i in max_pair_idx_list]
        max_pairs = sorted(max_pairs, key=lambda pair: pair[0])

        first_pair_indices = list(range(len(max_pairs)))
        second_pair_indices = list(range(1,len(max_pairs))) + [0]
        for pair_1, pair_2 in zip(first_pair_indices, second_pair_indices):
            line_1 = [poly[max_pairs[pair_1][0]], poly[max_pairs[pair_1][1]]]
            line_2 = [poly[max_pairs[pair_2][0]], poly[max_pairs[pair_2][1]]]
            corners.append(self.line_intersection(line_1, line_2))
        corners = sorted(corners, key=lambda pt: (pt[1],pt[0])) # sorted so that warpPerspective works ok
        return np.array(corners, dtype="float32")

    @staticmethod
    def line_intersection(line1, line2):
        # TODO: here in general
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    # FOR PLOTTING
    def plot_corners(self, img):
        plt.imshow(img)
        try:
            for corner in self.corners:
                plt.scatter(corner[1], corner[0])
        except:
            print(len(self.corners))
        plt.show()

    def track_ball(self, img, color_thresh):
        """
        
        :param img: the image where the ball is being tracked 
        :param color_thresh: a list of two tuples, first is for the a-space thresholds, second is for b-space threshold
        :return: the location of the ball as a tuple
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        a_img = self.color_mask(lab, color_thresh[0], "a")
        b_img = self.color_mask(lab, color_thresh[1], "b")
        bin_img = np.logical_and(a_img, b_img).astype("uint8")
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        bin_img = cv2.erode(bin_img, kernel, 1)
        bin_img = cv2.dilate(bin_img, kernel, 1)

        cnts = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # # only proceed if the radius meets a minimum size
            # if radius > 10:
            #     # draw the circle and centroid on the frame,
            #     # then update the list of tracked points
            #     cv2.circle(img_copy, (int(x), int(y)), int(radius),
            #                (0, 255, 255), 2)
            #     cv2.circle(img_copy, center, 5, (0, 0, 255), -1)

        return center

    @staticmethod
    def plot_mat(data):
        # TODO: can get rid of this at some point
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(data, interpolation='nearest')
        fig.colorbar(cax)
        fig.show()

    # FOR PRESENTATION
    @staticmethod
    def save_mat(data, path, cmap):
        # TODO: can get rid of this at some point
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(data, interpolation='nearest', cmap=cmap)
        fig.colorbar(cax)
        fig.savefig(path)

    @staticmethod
    def save_im(data, path, cmap):
        # TODO: can get rid of this at some point
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(data, interpolation='nearest', cmap=cmap)
        ax.axis('off')
        fig.savefig(path)

    def steps_and_plots(self, img):
        out_path = os.path.join("images","for_slides")
        # Lab Explanation
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        for channel in range(3):
            self.save_im(lab[:, :, channel], os.path.join(out_path,"lab_"+str(channel)+".jpg"),"viridis")

        a_img = self.ab_threshold(lab, "a")
        self.save_im(a_img, os.path.join(out_path,"aimg.jpg"), "plasma")
        b_img = self.ab_threshold(lab, "b")
        self.save_im(b_img, os.path.join(out_path, "bimg.jpg"), "plasma")
        ab_img = np.logical_and(a_img, b_img).astype("uint8")
        self.save_im(ab_img, os.path.join(out_path, "abimg.jpg"), "plasma")

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        opening = cv2.morphologyEx(ab_img, cv2.MORPH_OPEN, kernel, iterations=self.morph_iter)
        self.save_im(opening, os.path.join(out_path, "opening.jpg"), "inferno")
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iter*3)
        self.save_im(closing, os.path.join(out_path, "closing.jpg"), "inferno")

        _, contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_copy = img.copy()
        cv2.drawContours(img_copy, contours, -1, (0, 0, 255), 3)
        self.save_im(img_copy, os.path.join(out_path, "contours.jpg"), "inferno")

        hull = cv2.convexHull(contours[0])
        cv2.drawContours(img_copy, [hull], -1, (255, 0, 0), 3)
        self.save_im(img_copy, os.path.join(out_path, "hull.jpg"), "inferno")
        poly = cv2.approxPolyDP(hull, 0.01*cv2.arcLength(hull, True), True)
        assert poly.shape[0] == 8
        cv2.drawContours(img_copy, [poly], -1, (0, 255, 0), 3)
        self.save_im(img_copy, os.path.join(out_path, "poly.jpg"), "inferno")

        poly = np.squeeze(poly, 1)
        # plt.imshow(img)
        self.corners = self.get_corners_from_poly(poly)
        self.warped = self.warp_to_rect(img)
        plt.imshow(self.warped)
        plt.savefig(os.path.join(out_path, "warped.jpg"))

        for corner in self.corners:
            plt.scatter(corner[0], corner[1], color="r")
        plt.savefig(os.path.join(out_path, "corners.jpg"))



