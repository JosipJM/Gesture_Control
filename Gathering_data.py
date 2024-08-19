#import time
import string
#import mediapipe as mp
#import cv2
from pathlib import Path
#import math
#
#import tensorflow
#
#import numpy as np
#import pyautogui
#import PIL
#
#folder= "Data/A"
#mp_drawing = mp.solutions.drawing_utils
#mp_hands = mp.solutions.hands
#hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
#cap = cv2.VideoCapture(0)
#alphabet = list(map(chr, range(65, 91)))
#print(alphabet)
#c_alphabet=-1
#c_frame=100
#while cap.isOpened():
#    ret, frame = cap.read()
#    frame = cv2.flip(frame, 1)
#
#    if ret:
#        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#        result = hands.process(RGB_frame)
#
#
#        if result.multi_hand_landmarks:
#            for num, hand_landmarks in enumerate(result.multi_hand_landmarks):
#                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#
#
#                result = hands.process(RGB_frame)
#                if result.multi_hand_landmarks:
#                    for handType, handLms in zip(result.multi_handedness, result.multi_hand_landmarks):
#                        myHand = {}
#                        mylmList = []
#                        xList = []
#                        yList = []
#                        for id, lm in enumerate(handLms.landmark):
#                            # Scale the landmark positions to fit the screen size
#                            px, py = int(lm.x * 640), int(lm.y * 480)
#                            mylmList.append([px, py])
#                            xList.append(px)
#                            yList.append(py)
#
#                        # Calculate bounding box for the hand
#                        xmin, xmax = min(xList), max(xList)
#                        ymin, ymax = min(yList), max(yList)
#
#                        # Example: You could store these values or use them as needed
#                        myHand['lmList'] = mylmList
#                        myHand['bbox'] = (xmin, ymin, xmax, ymax)
#                        # You can also use the bounding box to draw a rectangle around the hand, etc.
#
#                        # Example usage: Draw bounding box on the frame (optional)
#
#                    white_image=np.ones((300,300,3), dtype=np.uint8)*255
#                    Volume_frame = frame[ymin-20:ymax+20, xmin-20:xmax+20]
#                    Frame_shape=Volume_frame.shape
#                    ratio=(ymax-ymin+40)/(xmax-xmin+40)
#                    if ratio > 1:
#                        k = 300/(ymax-ymin+40)
#                        wCal =math.ceil(k* (-xmin+20+xmax+20))
#                        wCal=min(wCal,300)
#
#                        imgResize = cv2.resize(Volume_frame,(wCal,300))
#                        Volime_reshape=imgResize.shape
#                        wGap = math.ceil((300-wCal)/2)
#                        white_image[0:Volime_reshape[0],wGap:Volime_reshape[1]+wGap]=imgResize
#                    else:
#                        k = 300 / (xmax - xmin + 40)
#                        wCal = math.ceil(k * (-ymin + 20 + ymax + 20))
#                        wCal=min(wCal,300)
#                        imgResize = cv2.resize(Volume_frame, (300,wCal))
#                        Volime_reshape = imgResize.shape
#                        wGap = math.ceil((300 - wCal) / 2)
#                        white_image[wGap:Volime_reshape[0]+ wGap, 0:Volime_reshape[1] ] = imgResize
#
#                        pass
#                    try:
#                        cv2.imshow('Volume frame', Volume_frame)
#                        cv2.imshow('White', white_image)
#
#                    except:
#                        pass
#                    cv2.imshow("Camera", frame)
#
#                    if cv2.waitKey(1) & 0xFF == ord('q'):
#                        break
#                    if c_alphabet < 26:
#                        if c_frame < 100:
#                            if cv2.waitKey(1) == ord(alphabet[c_alphabet]):
#                                print(alphabet[c_alphabet])
#                                cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', Volume_frame)
#                                c_frame+=1
#                        else:
#                            c_alphabet+=1
#
#                            c_frame=0
#                            segments = folder.split('/')
#                            last_segment = segments.pop()  # This pops the last element ("B") from the list
#                            remaining_folder = '/'.join(segments)
#                            folder = remaining_folder+"/" + alphabet[c_alphabet]
#                            folder_path = Path(folder)
#                            folder_path.mkdir(parents=True, exist_ok=True)
#                            print(folder_path)
#
#cap.release()
#cv2.destroyAllWindows()
#
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "data/M"
counter = 0
c_alphabet=12
c_frame=100
alphabet = list(map(chr, range(65, 91)))
print(alphabet)
while True:
    success, img = cap.read()
    cv2.flip(img,1,img)
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
    cv2.imshow("ImageCrop", imgCrop)
    cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img)
    print("all good")
    print(alphabet[c_alphabet])
    if cv2.waitKey(1)== ord(alphabet[c_alphabet]):
        print("ok")
        if c_alphabet <26:
            if c_frame <400:
                c_frame += 1
                cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
                print(c_frame)
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