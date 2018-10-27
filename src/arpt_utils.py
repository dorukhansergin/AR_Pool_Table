import cv2
import numpy as np
import matplotlib.pyplot as plt


class ARPoolTable:
    def __init__(self, a_thresh, b_thresh, ero_dil_num_iter, kernel_size):
        # TODO: docstring
        self.a_thresh = a_thresh
        self.b_thresh = b_thresh
        self.ero_dil_num_iter = ero_dil_num_iter
        self.kernel_size = kernel_size
        self.corners = None

    def table_detector(self, img):
        # TODO: docstring
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        a_img = self.ab_threshold(lab, "a")
        b_img = self.ab_threshold(lab, "b")
        ab_img = np.logical_and(a_img, b_img).astype("uint8")
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        ab_img = cv2.erode(ab_img, kernel, iterations=self.ero_dil_num_iter)
        # ab_img = cv2.dilate(ab_img, kernel, iterations=self.ero_dil_num_iter)
        self.corners = self.get_corners(ab_img)

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

    @staticmethod
    def optimum_dilation_erosion_kernel_size(contour_area):
        # TODO: figure out here
        return 15

    @staticmethod
    def get_corners(bin_img):
        (x, y) = np.where(bin_img)
        corners = [[x.min(),y.min()],[x.max(),y.min()],[x.max(),y.max()],[x.min(),y.max()]]
        return corners

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

        return center

    @staticmethod
    def plot_mat(data):
        # TODO: can get rid of this at some point
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(data, interpolation='nearest')
        fig.colorbar(cax)
        fig.show()
