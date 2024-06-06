import cv2
import numpy as np

def get_ball_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 1
    params.maxThreshold = 256
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 2000
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    return detector 

def get_robot_circle_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 1
    params.maxThreshold = 256
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 600
    params.maxArea = 10000000
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = False
    # Create a detector with the parameters
    return cv2.SimpleBlobDetector_create(params)
