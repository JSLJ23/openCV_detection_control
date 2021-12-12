import cv2
import mediapipe as mpi
import time
import math
import numpy as np


class handDetector():
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_confidence=0.5, tracking_confidence=0.5,
                 id_list=None):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.id_list = id_list

        self.mpHands = mpi.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.model_complexity,
                                        self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mpi.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hands_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hands_landmarks, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_number=1, draw=True):

        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number - 1]
            for id, lm in enumerate(my_hand.landmark):
                height, width, channels = img.shape
                center_x, center_y = int(lm.x * width), int(lm.y * height)
                lm_list.append([id, center_x, center_y])
                if draw and id in self.id_list:
                    cv2.circle(img, (center_x, center_y), 10, (255, 0, 255), cv2.FILLED)

        return lm_list

    @staticmethod
    def fingers_up(lm_list):
        if len(lm_list) > 0:
            if lm_list[8][2] < lm_list[6][2]:
                pass

    @staticmethod
    def find_thumb_index_midpoint(lm_list, img):
        if len(lm_list) > 0:
            x_thumb, y_thumb = lm_list[4][1:3]
            x_index_finger, y_index_finger = lm_list[8][1:3]

            cv2.circle(img, (x_thumb, y_thumb), 15, (255, 0, 255), cv2.FILLED)  # Thumb circle
            cv2.circle(img, (x_index_finger, y_index_finger), 15, (255, 0, 255), cv2.FILLED)  # Index finger circle
            # Line between two circles of thumb and index finger
            cv2.line(img, (x_thumb, y_thumb), (x_index_finger, y_index_finger), (255, 0, 255), 3)

            length_of_line = math.hypot(x_index_finger - x_thumb, y_index_finger - y_thumb)

            if length_of_line < 30:
                cv2.circle(img, (x_index_finger, y_index_finger), 15, (0, 255, 0), cv2.FILLED)

        else:
            x_index_finger, y_index_finger = None, None
            length_of_line = None

        return length_of_line, x_index_finger, y_index_finger

    @staticmethod
    def get_angles(lm_list, fingers):

        angle_degrees = 0

        if len(lm_list) > 0:

            a = np.array(lm_list[fingers[0]][1:])
            b = np.array(lm_list[fingers[1]][1:])
            c = np.array(lm_list[fingers[2]][1:])

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)

            angle_degrees = np.degrees(angle)

        return angle_degrees


def main():
    previous_time = 0

    id_list = [2, 3, 4, 5, 6, 7, 8]

    cap = cv2.VideoCapture(0)
    detector = handDetector(id_list=id_list)

    while True:
        success, img = cap.read()
        img_2 = detector.find_hands(img)
        lm_list_2 = detector.find_position(img)
        if len(lm_list_2) != 0:
            print(lm_list_2[4], lm_list_2[8])

        # create frames per second (fps) variable
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img_2, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Camera feed", img_2)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
