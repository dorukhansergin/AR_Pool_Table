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
    def __init__(self):
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
            ret, frame = cap.read()
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            pt_corners = self.detectCorners(frame)
            if pt_corners is not None:
                warp = self.warpToRect(frame, pt_corners)
                cv2.imshow('frame', warp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        cap.release()
        cv2.destroyAllWindows()