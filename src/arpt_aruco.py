import os
import copy
import functools
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

arc_len_partial = functools.partial(cv2.arcLength, closed=True)

#TODO: delete prints

class ARPT:
    def __init__(self, a_thresh, b_thresh):
        self.a_thresh = a_thresh
        self.b_thresh = b_thresh
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters_create()
        self.cue_center = None
        self.cue_radius = None
        self.cue_track = []
        self.corners = None
        self.warp = None
        self.frame_height = None
        self.frame_width = None
        self.old_cue_center = None

    def detectCorners(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        if ids is None:
            return False
        ids = [id[0] for id in ids]
        pt_corners = []
        for id in id_order:
            try:
                idx = ids.index(id)
            except:
                return True
            else:
                pt_corners.append(corners[idx][:,aruco_corner_match[id],:])
        self.corners = np.vstack(pt_corners)
        return True

    def detectCornersFirst(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        if ids is None:
            return False
        ids = [id[0] for id in ids]
        pt_corners = []
        for id in id_order:
            try:
                idx = ids.index(id)
            except:
                return False
            else:
                pt_corners.append(corners[idx][:,aruco_corner_match[id],:])
        self.corners = np.vstack(pt_corners)
        return True

    def warpToRect(self, frame):
        # code taken from https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
        self.frame_height, self.frame_width = frame.shape[0], frame.shape[1]
        dst = np.array([
            [0, 0],
            [self.frame_width  - 1, 0],
            [0, self.frame_height - 1],
            [self.frame_width  - 1, self.frame_height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(self.corners.astype(np.float32), dst)
        warp = cv2.warpPerspective(frame, M, (self.frame_width, self.frame_height))
        self.warp = warp

    def warpBack(self):
        src = np.array([
            [0, 0],
            [self.frame_width  - 1, 0],
            [0, self.frame_height - 1],
            [self.frame_width  - 1, self.frame_height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src, self.corners.astype(np.float32))
        orig = cv2.warpPerspective(self.warp, M, (self.frame_width, self.frame_height))
        return orig

    def runVideo(self, video_path):
        # Get the first corner
        cap = cv2.VideoCapture(video_path)
        while (True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            corners_found = self.detectCornersFirst(frame)
            if corners_found:
                break
        cap.release()

        # Now you have it you can go on running the video
        cap = cv2.VideoCapture(video_path)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))/2)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, size)

        while(True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            corners_found = self.detectCorners(frame)
            if corners_found:
                self.warpToRect(frame)
                self.tableSegmentation()
                if self.cue_radius > 30:
                    self.trackCueBall()
                frame = self.warpBack()
                out.write(frame)
                cv2.imshow('frame', frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def tableSegmentation(self):
        # Convert Color to Lab space
        lab = cv2.cvtColor(self.warp, cv2.COLOR_BGR2Lab)

        # Color threshold on a and b colorspaces, then get an elementwise AND of both
        a_img = self.ab_threshold(lab, "a")
        b_img = self.ab_threshold(lab, "b")
        ab_img = np.logical_and(a_img, b_img).astype("uint8")

        self.detectCueBall(self.warp, ab_img)
        print(self.cue_radius > 28, self.ballMoving() )
        if self.cue_radius > 28 and not self.ballMovingFast():
            self.checkCue()

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
        if self.old_cue_center is None:
            self.old_cue_center = (int(x), int(y))
        else:
            self.old_cue_center = copy.deepcopy(self.cue_center)
        self.cue_center, self.cue_radius = (int(x), int(y)), int(radius)
        if cv2.norm(self.old_cue_center, self.cue_center) > 10:
            self.cue_track.append(self.cue_center)

    def isCircle(self, cnt):
        arcl = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if area <= 0:
            return False
        else:
            circle_prop = arcl ** 2 / area
            return 10 < circle_prop < 20

    def ballMoving(self):
        if len(self.cue_track) > 3:
            if cv2.norm(self.cue_track[-1], self.cue_track[-2]) < 10:
                return False
            else:
                return True
        else:
            return False

    def ballMovingFast(self):
        if len(self.cue_track) > 3:
            if cv2.norm(self.cue_track[-1], self.cue_track[-2]) < 200:
                return False
            else:
                return True
        else:
            return False

    def trackCueBall(self):
        """ code inspired from https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/"""
        for i in range(len(self.cue_track)):
            if self.cue_track[i - 1] is None or self.cue_track[i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            # TODO: the following is not a robust solution, find a better one
            if cv2.norm(self.cue_track[i - 1], self.cue_track[i]) < 100:
                cv2.line(self.warp, self.cue_track[i - 1], self.cue_track[i], (0, 0, 255), thickness)

    def checkCue(self):
        # 1) Crop out cue ball surrounding
        scaling_factor = 2
        y = self.cue_center[1]
        x = self.cue_center[0]
        mini = self.warp[y - scaling_factor * self.cue_radius:y + scaling_factor * self.cue_radius,
                         x - scaling_factor * self.cue_radius:x + scaling_factor * self.cue_radius, :]

        # 2) True or False, whether cue is in there or not?
        lab = cv2.cvtColor(mini, cv2.COLOR_BGR2Lab)
        a_img = self.ab_threshold(lab, "a")
        b_img = self.ab_threshold(lab, "b")
        ab_img = np.logical_and(a_img, b_img).astype("uint8")

        # close the img
        inv_img = ~ab_img

        # perform morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(inv_img, cv2.MORPH_OPEN, kernel, iterations=3)

        # draw the contours on the opened image
        pad = 1
        constant = cv2.copyMakeBorder(opening, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0])
        _, contours, _ = cv2.findContours(~constant, cv2.RETR_TREE, cv2.cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 3:
            return False
        else:
            print("HERE")
            cv2.drawContours(mini, contours, -1, (0, 0, 255), 3)
            sort_contours = sorted(contours, key=arc_len_partial)
            # eliminates the contour corresponding to the border
            sort_contours.pop()
            # eliminate contour corresponding to ball
            for contour in sort_contours:
                is_ball = cv2.pointPolygonTest(contour, (mini.shape[0] // 2, mini.shape[1] // 2), False)
                if is_ball != 1:
                    stick_contour = contour
                    continue
            # try catch this in implementation
            # CENTER IN MINI
            center_mini = (mini.shape[0]/2, mini.shape[1]/2)
            # cv2.circle(mini, center_mini, 3, (255,0,0), -1)
            # CENTER OF RECT
            cv2.drawContours(mini, [stick_contour], -1, (0, 0, 255), 3)
            rect = cv2.minAreaRect(stick_contour)
            rect_center = (int(rect[0][0]), int(rect[0][1]))
            actual_rect_center = np.array(rect_center) + self.cue_center - np.array(center_mini)
            actual_rect_center = (actual_rect_center[0], actual_rect_center[1])
            edgePoint = self.getEdgePoint(actual_rect_center)
            edgePoint = (int(edgePoint[0]), int(edgePoint[1]))
            if self.isInPocket(edgePoint):
                line_color = (0, 255, 0)
            else:
                line_color = (255,0,0)
            cv2.circle(self.warp, actual_rect_center, 7, (0, 255, 0), -1)
            cv2.circle(self.warp, self.cue_center, 7, (0, 255, 0), -1)
            cv2.line(self.warp, self.cue_center, edgePoint, line_color, 10)


            # cv2.circle(mini, rect_center, 3, (255, 0, 0), -1)
            #
            # vec =  np.array(center_mini) - np.array(rect[0])
            # angle = np.arctan2(vec[0], vec[1])
            # angle = angle * 180 / np.pi
            # print(-rect[2], angle)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(mini, [box], 0, (0, 255, 255), 2)
        # If so, is it aligned with the cue ball or not

    def getEdgePoint(self, rect_center):
        y_delta = float(self.cue_center[1]) - float(rect_center[1])
        x_delta = float(self.cue_center[0]) - float(rect_center[0])
        m = y_delta / x_delta
        b = float(rect_center[1]) - m * float(rect_center[0])
        y_limit_max = self.cue_center[1] > rect_center[1]
        x_limit_max = self.cue_center[0] > rect_center[0]
        if y_limit_max and x_limit_max:
            # print("Case 1")
            y_max = m * self.warp.shape[0] + b
            x_max = (self.warp.shape[1] - b) / m
            if y_max > self.warp.shape[1]:
                return x_max, self.warp.shape[1]
            else:
                return self.warp.shape[0], y_max
        elif y_limit_max and not x_limit_max:
            # print("Case 2")
            y_max = m * 0 + b
            x_min = (self.warp.shape[1] - b) / m
            if y_max > self.warp.shape[1]:
                return x_min, self.warp.shape[1]
            else:
                return 0, y_max
        elif not y_limit_max and x_limit_max:
            # print("Case 3")
            y_min = m * self.warp.shape[0] + b
            x_max = (0 - b) / m
            if y_min < 0:
                return x_max, 0
            else:
                return self.warp.shape[0], y_min
        else:
            # print("Case 4")
            y_min = m * 0 + b
            x_min = (0 - b) / m
            # print(x_min, y_min)
            if y_min < 0:
                return x_min, 0
            else:
                return 0, y_min

    def isInPocket(self, edgePoint):
        if edgePoint[0] < 25 and edgePoint[1] < 25:
            return True
        else:
            return False

#### VIDEO OUTPUTTER
    def cornerTrackingOnActual(self, video_path):
        # Find the first corners
        cap = cv2.VideoCapture(video_path)
        while (True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            corners_found = self.detectCornersFirst(frame)
            if corners_found:
                break
        cap.release()

        cap = cv2.VideoCapture(video_path)

        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) / 2)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('corners.avi', fourcc, 20.0, size)

        while (True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            corners_found = self.detectCorners(frame)
            if corners_found:
                for corner in self.corners:
                    cv2.circle(frame, (corner[0], corner[1]), 8, (0, 0, 255), 5)
                out.write(frame)
                cv2.imshow('frame', frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def warpedVideo(self, video_path):
        # Get the first corner
        cap = cv2.VideoCapture(video_path)
        while (True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            corners_found = self.detectCornersFirst(frame)
            if corners_found:
                break
        cap.release()

        # Now you have it you can go on running the video
        cap = cv2.VideoCapture(video_path)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) / 2)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('warped.avi', fourcc, 20.0, size)

        while (True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            corners_found = self.detectCorners(frame)
            if corners_found:
                self.warpToRect(frame)
                out.write(self.warp)
                cv2.imshow('frame', self.warp)
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def shotTracking(self, video_path):
        # Get the first corner
        cap = cv2.VideoCapture(video_path)
        while (True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            corners_found = self.detectCornersFirst(frame)
            if corners_found:
                break
        cap.release()

        # Now you have it you can go on running the video
        cap = cv2.VideoCapture(video_path)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))/2)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('shot_tracking.avi', fourcc, 20.0, size)

        while(True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            corners_found = self.detectCorners(frame)
            if corners_found:
                self.warpToRect(frame)
                self.tableSegmentation()
                frame = self.warpBack()
                out.write(frame)
                cv2.imshow('frame', frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def ballTracking(self, video_path):
        # Get the first corner
        cap = cv2.VideoCapture(video_path)
        while (True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            corners_found = self.detectCornersFirst(frame)
            if corners_found:
                break
        cap.release()

        # Now you have it you can go on running the video
        cap = cv2.VideoCapture(video_path)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))/2)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('ball_tracking.avi', fourcc, 20.0, size)

        while(True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            corners_found = self.detectCorners(frame)
            if corners_found:
                self.warpToRect(frame)
                self.tableSegmentation()
                # TODO: is the following if a robust solution??
                print(self.cue_radius)
                if self.cue_radius > 30 and self.ballMoving():
                    self.trackCueBall()
                frame = self.warpBack()
                out.write(frame)
                cv2.imshow('frame', frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def contourTracking(self, video_path):
        # Get the first corner
        cap = cv2.VideoCapture(video_path)
        while (True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            corners_found = self.detectCornersFirst(frame)
            if corners_found:
                break
        cap.release()

        # Now you have it you can go on running the video
        cap = cv2.VideoCapture(video_path)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) / 2)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('contour_tracking.avi', fourcc, 20.0, size)

        while(True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            corners_found = self.detectCorners(frame)
            if corners_found:
                self.warpToRect(frame)
                # Convert Color to Lab space
                lab = cv2.cvtColor(self.warp, cv2.COLOR_BGR2Lab)

                # Color threshold on a and b colorspaces, then get an elementwise AND of both
                a_img = self.ab_threshold(lab, "a")
                b_img = self.ab_threshold(lab, "b")
                ab_img = np.logical_and(a_img, b_img).astype("uint8")

                self.detectCueBall(self.warp, ab_img)
                if self.cue_radius > 28 and not self.ballMoving():
                    self.checkCue()

                out.write(frame)
                cv2.imshow('frame', frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

