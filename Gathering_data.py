import time
import mediapipe as mp
import cv2
from pathlib import Path

import numpy as np


folder= "Data/A"
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
alphabet = list(map(chr, range(65, 91)))
print(alphabet)
c_alphabet=-1
c_frame=100
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(RGB_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks, hand_type in zip(result.multi_hand_landmarks, result.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            lm_list = [(int(lm.x * 640), int(lm.y * 480)) for lm in hand_landmarks.landmark]
            x_list, y_list = zip(*lm_list)

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            # Extract the hand region and prepare the white image
            hand_region = frame[max(0, ymin-20):min(480, ymax+20), max(0, xmin-20):min(640, xmax+20)]
            white_image = np.ones((300, 300, 3), dtype=np.uint8) * 255

            h, w = ymax - ymin + 40, xmax - xmin + 40
            if h > w:
                scale = 300 / h
                new_w = min(int(scale * w), 300)
                resized = cv2.resize(hand_region, (new_w, 300))
                x_offset = (300 - new_w) // 2
                white_image[:, x_offset:x_offset + new_w] = resized
            else:
                scale = 300 / w
                new_h = min(int(scale * h), 300)
                resized = cv2.resize(hand_region, (300, new_h))
                y_offset = (300 - new_h) // 2
                white_image[y_offset:y_offset + new_h, :] = resized

            # Display the results
            cv2.imshow('Hand Region', white_image)
            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print(alphabet[c_alphabet])
            if c_alphabet < 26:
                if c_frame < 100:
                    if cv2.waitKey(1) == ord(alphabet[c_alphabet]):
                        print(alphabet[c_alphabet])
                        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', white_image)
                        c_frame+=1
                else:
                    c_alphabet+=1

                    c_frame=0
                    segments = folder.split('/')
                    last_segment = segments.pop()  # This pops the last element ("B") from the list
                    remaining_folder = '/'.join(segments)
                    folder = remaining_folder+"/" + alphabet[c_alphabet]
                    folder_path = Path(folder)
                    folder_path.mkdir(parents=True, exist_ok=True)
                    print(folder_path)

cap.release()
cv2.destroyAllWindows()
