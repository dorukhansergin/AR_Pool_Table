import os
import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt

aruco_corner_match = {
    6: 2,
    3: 0,
    2: 1,
    7: 3
}

id_order = [3,7,6,2]


class ARPT:
    def __init__(self, a_thresh, b_thresh):
        self.a_thresh = a_thresh
        self.b_thresh = b_thresh
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters_create()

    def detectCorners(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        if ids is None:
            return None
        ids = [id[0] for id in ids]
        pt_corners = []
        for id in id_order:
            try:
                idx = ids.index(id)
            except:
                return None
            else:
                pt_corners.append(corners[idx][:,aruco_corner_match[id],:])
        return np.vstack(pt_corners)

    def warpToRect(self, frame, corners):
        # code taken from https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
        maxHeight, maxWidth = frame.shape[0], frame.shape[1]

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [0, maxHeight - 1],
            [maxWidth - 1, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        warp = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
        return warp

    def runVideo(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while(cap.isOpened()):
            go_break = False # TODO: delete
            ret, frame = cap.read()
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

            pt_corners = self.detectCorners(frame)
            if pt_corners is not None:
                go_break = True
                warp = self.warpToRect(frame, pt_corners)
                # lab = cv2.cvtColor(warp, cv2.COLOR_BGR2Lab)
                lab = self.tableSegmentation(warp)
                cv2.imshow('frame', lab)
                # plt.matshow(lab)
                # plt.show()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # TODO: delete
            # if go_break:
            #     break


        cap.release()
        cv2.destroyAllWindows()

    def tableSegmentation(self, warp):
        # Convert Color to Lab space
        lab = cv2.cvtColor(warp, cv2.COLOR_BGR2Lab)

        # Color threshold on a and b colorspaces, then get an elementwise AND of both
        a_img = self.ab_threshold(lab, "a")
        b_img = self.ab_threshold(lab, "b")
        ab_img = np.logical_and(a_img, b_img).astype("uint8")

        warp = self.detectBalls(warp, ab_img)
        return warp
        # # Apply opening and closing operations one after another
        # kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        # opening = cv2.morphologyEx(ab_img, cv2.MORPH_OPEN, kernel, iterations=self.morph_iter)
        # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iter*3)
        #
        # # Find the convex hull of the contour and then approximate a polygon around it
        # im2, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # assert len(contours) == 1 # make sure we have a single component
        # hull = cv2.convexHull(contours[0])
        # poly = cv2.approxPolyDP(hull, 0.01*cv2.arcLength(hull,True), True)
        # assert poly.shape[0] == 8 # make sure we get 4 corners

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

    def detectCueBall(self, img, bin):
        """
        Returns the locations and the radi of the balls detected on the pool table

        :param bin: A binary image where 1s are table 0s are everything else
        :return:
        """
        _, cnts, _ = cv2.findContours(bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cue_ball_cnt = max(filter(lambda x: self.isCircle(x), cnts), key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(cue_ball_cnt)
        return (x, y), radius

    def isCircle(self, cnt):
        arcl = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if area <= 0:
            return False
        else:
            circle_prop = arcl ** 2 / area
            return 10 < circle_prop < 20

