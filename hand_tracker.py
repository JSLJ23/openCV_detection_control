import cv2
import mediapipe as mpi
import time

cap = cv2.VideoCapture(0)

mpHands = mpi.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mpi.solutions.drawing_utils

previous_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hands_landmarks in results.multi_hand_landmarks:

            mpDraw.draw_landmarks(img, hands_landmarks)  # mpHands.HAND_CONNECTIONS

    # create frames per second (fps) variable
    # current_time = time.time()
    # fps = 1/(current_time - previous_time)
    # previous_time = current_time

    # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
