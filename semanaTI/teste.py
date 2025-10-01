import cv2
import mediapipe as mp



    # pra fechar a janela do video é sá dar um crtl + c no terminal :)




video = cv2.VideoCapture(0)

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils




while True:
    check, img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = Hand.process(imgRGB)
    handsPoints = result.multi_hand_landmarks
    h, w, _ = img.shape
    pontos = []

    if handsPoints:
        for points in handsPoints:

            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                #cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                pontos.append((cx, cy))
        
        dedos = [8, 12, 16, 20]
        contador = 0
        if points:
            for x in dedos:
                if pontos[x][1] < pontos[x - 2][1]:
                    contador += 1
        print(contador)     

        if contador == 2:
            cv2.putText(
                        img,                                
                        f"COMANDO VERMELHO",    
                        (50, 50),                            
                        cv2.FONT_HERSHEY_SIMPLEX,            
                        1,                                   
                        (0, 0, 255),                         
                        2                                    
                    )      
        else: 
            cv2.putText(
                        img,                                
                        f"JURADO DE MORTE ",    
                        (50, 50),                            
                        cv2.FONT_HERSHEY_SIMPLEX,            
                        1,                                   
                        (0, 0, 255),                         
                        2                                    
                    )   
                   
        
    
    cv2.imshow("Video", img)
    cv2.waitKey(1)


