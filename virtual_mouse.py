import math
import cv2
import time
import numpy as np
import autopy

import hand_tracking_module as htm

############################################
# Camera display properties ###############
############################################
width_camera = 640
height_camera = 480
width_screen = 1920
height_screen = 1080
############################################

cap = cv2.VideoCapture(0)
cap.set(3, width_camera)
cap.set(4, height_camera)

detector = htm.handDetector(detection_confidence=0.7)

while True:
    success, img = cap.read()

    image = detector.find_hands(img, draw=True)
    lm_list = detector.find_position(img, draw=False)
    thumb_and_index, x_index_finger, y_index_finger = detector.find_thumb_index_midpoint(lm_list, image)
    thumb_angle_1 = detector.get_angles(lm_list, [4, 3, 2])
    thumb_angle_2 = detector.get_angles(lm_list, [3, 2, 1])
    thumb_angle_1_and_2 = thumb_angle_1 + thumb_angle_2

    x_rescaled = np.interp(x_index_finger, (0, width_camera), (0, width_screen))
    y_rescaled = np.interp(y_index_finger, (0, height_camera), (0, height_screen))

    if 0 < x_rescaled < width_screen and 0 < y_rescaled < height_screen:
        autopy.mouse.move(width_screen - x_rescaled, y_rescaled)  # Left to right orientation must be flipped

    print(thumb_angle_1_and_2)

    print(x_rescaled, y_rescaled)

    cv2.imshow("Camera feed", img)
    cv2.waitKey(1)

