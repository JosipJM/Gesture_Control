import mediapipe as mp
import tensorflow as tf
import cv2

import math
import numpy as np

import pyautogui
import autopy

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Load the pre-trained model
modelPath = "Model/keras_model.h5"
labelFile = "Model/labels.txt"
Model = tf.keras.models.load_model(modelPath)
Data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Initialize Audio
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Constants and Globals
pyautogui.FAILSAFE = False
tracker = np.zeros(20)
x_cord = np.zeros(21)
y_cord = np.zeros(21)
prev_right_x = [0, 0]
prev_right_y = [0, 0]
off_set = 20
mode = 0
prev_left_x, prev_left_y = 240, 240

# Load labels
if labelFile:
    with open(labelFile, "r") as f:
        list_labels = [line.strip() for line in f]
else:
    print("No Labels Found")


# Helper functions


def mode_change(joint1, joint2):
    global mode
    if joint1 - joint2 < 0 and tracker[3] == 0:
        tracker[3] = 1
        mode += 1
    elif joint1 - joint2 > 0 and tracker[3] == 1:
        tracker[3] = 0


def distance(p1, p2, p3, p4):
    return pow(pow(p1 - p2, 2) + pow(p3 - p4, 2), 0.5)


def get_finger_coords(hand, joint):
    index_tip = hand.landmark[joint]
    # *1.5 changes dpi and -400 offsetting minimum needed value to reach edge
    return int(index_tip.x * 1.5 * 1920 - 400), int(index_tip.y * 1.5 * 1080 - 400)


def click(joint1, joint2):
    if tracker[0] == 0 and joint1 - joint2 < 0:
        print("click")
        pyautogui.click()
        tracker[0] = 1

    elif tracker[0] == 1 and joint1 - joint2 > 0:
        tracker[0] = 0


def double_click(joint1, joint2):
    if tracker[1] == 0 and joint1 - joint2 < 0:
        print("double click")
        pyautogui.doubleClick()
        tracker[1] = 1

    elif tracker[1] == 1 and joint1 - joint2 > 0:
        tracker[1] = 0


def scroll(joint1, joint2):
    if joint2 - joint1 < 0:
        print(joint1, joint2)
        if joint2 < 640:
            print("scroll down")
            pyautogui.scroll(int((640 - joint1) * 0.5))
        else:
            print("scroll up")
            pyautogui.scroll(int((640 - joint1) * 0.5))


def left_click(joint1, joint2):
    if tracker[2] == 0 and joint1 - joint2 < 0:
        print("left click")
        pyautogui.click(button='right')
        tracker[2] = 1
    elif tracker[2] == 1 and joint1 - joint2 > 0:
        tracker[2] = 0


def back(joint1, joint2):
    if tracker[4] == 0 and joint1 - joint2 < 0:
        print("back")
        with pyautogui.hold('alt'):
            pyautogui.press(['left'])
        tracker[4] = 1
    if tracker[4] == 1 and joint1 - joint2 > 0:
        tracker[4] = 0


def forward(joint1, joint2):
    if tracker[5] == 0 and joint1 - joint2 < 0:
        print("forward")
        with pyautogui.hold('alt'):
            pyautogui.press(['right'])
        tracker[5] = 1
    if tracker[5] == 1 and joint1 - joint2 > 0:
        tracker[5] = 0


def hold_click(joint1, joint2):
    if joint1 - joint2 > 0 and tracker[6] == 1:
        print("hold")
        pyautogui.mouseUp()
        tracker[6] = 0
    elif joint1 - joint2 < 0 and tracker[6] == 0:
        print("let go")
        pyautogui.mouseDown()
        tracker[6] = 1


def enter(joint1, joint2):
    if tracker[12] == 0 and joint1 - joint2 < 0:
        pyautogui.press('enter')
        tracker[12] = 1
    if tracker[12] == 1 and joint1 - joint2 > 0:
        tracker[12] = 0


def key_press(joint1, joint2, key, num_tracker):
    if tracker[num_tracker] == 0 and joint1 - joint2 < 0:
        pyautogui.press(key)
        tracker[num_tracker] = 1
    if tracker[num_tracker] == 1 and joint1 - joint2 > 0:
        tracker[num_tracker] = 0


def get_hand_label_and_wrist_coords(index, hand, results):

    output = None
    if index == 0:
        label = results.multi_handedness[0].classification[0].label
        coords = tuple(np.multiply(
            np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640, 480]).astype(int))

        output = label, coords
        return output

    if index == 1:
        label = results.multi_handedness[1].classification[0].label
        coords = tuple(np.multiply(
            np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640, 480]).astype(int))

        output = label, coords
        return output
    return output, None


def smooth_movement(prev_coords, new_coords, smoothing_factor=0.7):
    smoothed_x = int(prev_coords[0] * smoothing_factor + new_coords[0] * (1 - smoothing_factor))
    smoothed_y = int(prev_coords[1] * smoothing_factor + new_coords[1] * (1 - smoothing_factor))
    return smoothed_x, smoothed_y


def move_mouse_safely(x, y):
    global screen_height, screen_width
    x = max(0, min(screen_width - 1, x))
    y = max(0, min(screen_height - 1, y))
    autopy.mouse.move(x, y)


def circle(img, joint_number, color, x_list, y_list):
    return cv2.circle(img, (x_list[joint_number], y_list[joint_number]), 5, color, -1)


# Video Capture Setup
cap = cv2.VideoCapture(0)

# Resize frame for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

screen_width, screen_height = autopy.screen.size()


def main():
    while cap.isOpened():
        # frame capturing
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        # Convert the frame to RGB for hand detection
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(RGB_frame)

        if result.multi_hand_landmarks and ret:
            for num, hand_landmarks in enumerate(result.multi_hand_landmarks):
                label = ""
                wrist_coords = 0
                try:
                    label, wrist_coords = get_hand_label_and_wrist_coords(num, hand_landmarks, result)
                except Exception as e:
                    print(f"Error adjusting volume: {e}")

                ##########################################################################
                ##########################################################################
                # ASL detection
                ##########################################################################
                ##########################################################################

                if label and wrist_coords and mode % 3 == 2:
                    cv2.putText(frame, "ASL detection ", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    try:
                        label, wrist_coords = get_hand_label_and_wrist_coords(num, hand_landmarks, result)
                    except Exception as e:
                        print(f"Error adjusting volume: {e}")

                    # Collect landmark positions
                    xList, yList = [], []
                    for lm in hand_landmarks.landmark:
                        px, py = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        xList.append(px)
                        yList.append(py)

                    # Calculate bounding box
                    x_min, x_max = min(xList), max(xList)
                    y_min, y_max = min(yList), max(yList)

                    # Create a white image to hold the processed hand image
                    white_image = np.ones((300, 300, 3), dtype=np.uint8) * 255

                    # Resize hand region
                    White_Frame = frame[y_min - 20:y_max + 20, x_min - 20:x_max + 20]
                    ratio = (y_max - y_min + 40) / (x_max - x_min + 40)

                    # Right hand functions

                    if label == "Right":
                        prev_right_y[0], prev_right_y[1] = yList[8], yList[7]

                        if xList[5] < xList[4]:
                            key_press(yList[11], yList[12], "left", 9)
                            key_press(yList[15], yList[16], "right", 10)

                            circle(frame, 11, (147, 81, 56), xList, yList)
                            circle(frame, 12, (147, 81, 56), xList, yList)

                            circle(frame, 15, (65, 235, 166), xList, yList)
                            circle(frame, 16, (65, 235, 166), xList, yList)

                            circle(frame, 4, (255, 160, 5), xList, yList)
                            circle(frame, 5, (255, 160, 5), xList, yList)

                            circle(frame, 20, (150, 255, 255), xList, yList)
                            circle(frame, 17, (150, 255, 255), xList, yList)

                            mode_change(yList[17], yList[20])
                        else:

                            key_press(yList[19], yList[20], "backspace", 11)
                            key_press(yList[11], yList[12], "enter", 12)

                            circle(frame, 11, (147, 81, 56), xList, yList)
                            circle(frame, 12, (147, 81, 56), xList, yList)

                            circle(frame, 8, (255, 160, 5), xList, yList)
                            circle(frame, 7, (255, 160, 5), xList, yList)

                            circle(frame, 20, (235, 15, 154), xList, yList)
                            circle(frame, 19, (235, 15, 154), xList, yList)

                    # Left hand functions

                    if label == "Left":
                        # white_image adjusting
                        try:
                            if ratio > 1:
                                k = 300 / (y_max - y_min + 40)
                                wCal = math.ceil(k * (x_max - x_min + 40))
                                imgResize = cv2.resize(White_Frame, (wCal, 300))
                                wGap = (300 - wCal) // 2
                                white_image[:, wGap:wGap + wCal] = imgResize
                            else:
                                k = 300 / (x_max - x_min + 40)
                                hCal = math.ceil(k * (y_max - y_min + 40))
                                imgResize = cv2.resize(White_Frame, (300, hCal))
                                hGap = (300 - hCal) // 2
                                white_image[hGap:hGap + hCal, :] = imgResize
                        except Exception as e:
                            print(f"Error: {e}")

                        cv2.imshow('White image', white_image)

                        # Prepare the white image for prediction
                        white_image_resized = cv2.resize(white_image, (224, 224))
                        frame_array = np.asarray(white_image_resized)
                        normalized_frame_array = (frame_array.astype(np.float32) / 127.0) - 1
                        Data[0] = normalized_frame_array
                        prediction = Model.predict(Data)
                        indexVal = np.argmax(prediction)
                        label = list_labels[indexVal]
                        label_split = label.split(" ")
                        last = label_split.pop()

                        # Display the prediction label
                        cv2.putText(frame, str(label), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        print(prev_right_y[1],prev_right_y[0])
                        if prev_right_y[1] < prev_right_y[0] and tracker[7] == 0 and prev_right_y[1] != 0 and prev_right_y[0] != 0:
                            pyautogui.press(last.lower())
                            tracker[7] = 1
                        elif prev_right_y[1] > prev_right_y[0] and tracker[7] == 1:
                            tracker[7] = 0

                ##########################################################################
                ##########################################################################
                # Volume adjustment
                ##########################################################################
                ##########################################################################

                if label and wrist_coords and mode % 3 == 1:
                    windows_volume = int(volume.GetMasterVolumeLevelScalar() * 100)
                    cv2.putText(frame, "Volume: {}".format(windows_volume), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, label, wrist_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    xList, yList = [], []
                    for lm in hand_landmarks.landmark:
                        px, py = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        xList.append(px)
                        yList.append(py)

                    # Left hand functions

                    if label == "Left":
                        circle(frame, 8, (200, 213, 48), xList, yList)
                        points_distance = int(distance(prev_right_x[0], xList[8], prev_right_y[0], yList[8]))
                        points_distance = max(120, min(points_distance, 350))
                        vol = np.interp(points_distance, [120, 350], [-96, 0])
                        if yList[18] < yList[20]:
                            try:
                                volume.SetMasterVolumeLevel(vol, None)
                            except Exception as e:
                                print(f"Error adjusting volume: {e}")
                        circle(frame, 20, (200, 213, 48), xList, yList)
                        circle(frame, 18, (200, 213, 48), xList, yList)
                        type(prev_right_x[0])
                        cv2.line(frame, (int(prev_right_x[0]), int(prev_right_y[0])), (xList[8], yList[8]),
                                 (200, 213, 48), 2, cv2.LINE_AA)

                    # Right hand functions

                    if label == "Right":
                        circle(frame, 8, (200, 213, 48), xList, yList)
                        if xList[5] < xList[4]:
                            mode_change(yList[17], yList[20])
                            circle(frame, 5, (255, 160, 5), xList, yList)

                            circle(frame, 20, (200, 213, 48), xList, yList)
                            circle(frame, 17, (200, 213, 48), xList, yList)
                        circle(frame, 4, (255, 160, 5), xList, yList)

                        prev_right_x[0], prev_right_y[0] = xList[8], yList[8]

                ##########################################################################
                ##########################################################################
                # Mouse control
                ##########################################################################
                ##########################################################################
                if label and wrist_coords and mode % 3 == 0:
                    cv2.putText(frame, "Mouse control mode", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    cv2.putText(frame, label, wrist_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)
                    xList, yList = [], []
                    for i in range(21):
                        # Getting hand coordinates for mouse movement
                        x_cord[i], y_cord[i] = get_finger_coords(hand_landmarks, i)
                        x_cord[i], y_cord[i] = int(x_cord[i]), int(y_cord[i])
                        px, py = int(hand_landmarks.landmark[i].x * frame.shape[1]), int(
                            hand_landmarks.landmark[i].y * frame.shape[0])

                        xList.append(px)
                        yList.append(py)

                    # Left hand functions

                    if label == "Left":
                        global prev_left_x, prev_left_y
                        if prev_left_x != x_cord[0] or prev_left_y != y_cord[0]:
                            smoothed_coords = smooth_movement((prev_left_x, prev_left_y), (x_cord[8], y_cord[8]))
                            move_mouse_safely(smoothed_coords[0], smoothed_coords[1])
                            prev_left_x, prev_left_y = smoothed_coords

                    # Right hand functions

                    if label == "Right":

                        if xList[4] - xList[5] < 0:
                            click(yList[7], yList[8])

                            double_click(yList[11], yList[12])

                            left_click(yList[19], yList[20])

                            hold_click(yList[15], yList[16])

                            circle(frame, 8, (255, 160, 5), xList, yList)
                            circle(frame, 7, (255, 150, 5), xList, yList)

                            circle(frame, 11, (100, 255, 80), xList, yList)
                            circle(frame, 12, (100, 255, 80), xList, yList)

                            circle(frame, 20, (200, 213, 48), xList, yList)
                            circle(frame, 19, (200, 213, 48), xList, yList)

                            circle(frame, 16, (168, 156, 205), xList, yList)
                            circle(frame, 15, (168, 156, 205), xList, yList)

                        else:
                            back(yList[7], yList[8])

                            forward(yList[11], yList[12])

                            scroll(y_cord[16], y_cord[15])

                            mode_change(yList[17], yList[20])

                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                            circle(frame, 5, (255, 160, 5), xList, yList)
                            circle(frame, 4, (255, 160, 5), xList, yList)
                            circle(frame, 8, (255, 160, 5), xList, yList)
                            circle(frame, 7, (255, 160, 5), xList, yList)

                            circle(frame, 11, (100, 255, 80), xList, yList)
                            circle(frame, 12, (100, 255, 80), xList, yList)

                            circle(frame, 20, (200, 213, 48), xList, yList)
                            circle(frame, 17, (200, 213, 48), xList, yList)

                            circle(frame, 16, (168, 156, 205), xList, yList)
                            circle(frame, 15, (168, 156, 205), xList, yList)

                        prev_right_x[0], prev_right_x[1] = x_cord[4], x_cord[8]

        # Display the camera frame
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
