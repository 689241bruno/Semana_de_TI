import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = Hand.process(imgRGB)
    handsPoints = result.multi_hand_landmarks

    if handsPoints:
        for points in handsPoints:
            print(points)
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)

    cv2.imshow("Video", img)
    cv2.waitKey(1)


    # pra fechar a janela do video é sá dar um crtl + c no terminal :)