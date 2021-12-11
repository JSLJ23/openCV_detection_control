import math
import cv2
import time
import numpy as np

import hand_tracking_module as htm

############################################
# Camera display properties ###############
############################################
width_camera = 640
height_camera = 480
############################################

cap = cv2.VideoCapture(0)
cap.set(3, width_camera)
cap.set(4, height_camera)

previous_time = 0

detector = htm.handDetector(detection_confidence=0.7)

while True:
    success, img = cap.read()

    image = detector.find_hands(img, draw=True)

    lm_list = detector.find_position(img, draw=False)
    # Thumb tip index = 4 and index finger tip index = 8
    # lm_list is a nested list of all 21 hand landmark lists containing the landmark index, x and y pixel coordinates
    if len(lm_list) > 0:
        print(lm_list[4], lm_list[8])

        # indexes the last two elements of landmark list excluding index 3
        x_thumb, y_thumb = lm_list[4][1:3]
        x_index_finger, y_index_finger = lm_list[8][1:3]

        cv2.circle(image, (x_thumb, y_thumb), 15, (255, 0, 255), cv2.FILLED)  # Thumb circle
        cv2.circle(image, (x_index_finger, y_index_finger), 15, (255, 0, 255), cv2.FILLED)  # Index finger circle
        # Line between two circles of thumb and index finger
        cv2.line(image, (x_thumb, y_thumb), (x_index_finger, y_index_finger), (255, 0, 255), 3)
        # Center of line and circle at the center of line
        x_center, y_center = (x_thumb + x_index_finger) // 2, (y_thumb + y_index_finger) // 2
        cv2.circle(image, (x_center, y_center), 15, (255, 0, 255), cv2.FILLED)

        length_of_line = math.hypot(x_index_finger-x_thumb, y_index_finger-y_thumb)
        print(length_of_line)

        if length_of_line < 50:



    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    cv2.putText(image, f"FPS {int(fps)}", (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Camera feed", img)
    cv2.waitKey(1)

