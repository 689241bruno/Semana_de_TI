import cv2
import mediapipe as mp
import csv
import os
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


CSV_FILE = 'dados_coletados.csv'

LIMITE_AMOSTRAS = 600 # frames de cada sinal pra mandar pra IA aprender


def criar_cabecalho_csv():
    if not os.path.exists(CSV_FILE):
        header = ['Sinal'] # letra q vc estiver fazendo 
        for i in range(21): # 21 pq são o total de pontos na mão
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        
        with open(CSV_FILE, mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(header)
        print(f"Arquivo CSV '{CSV_FILE}' criado com o cabeçalho.")
    else:
        print(f"Arquivo CSV '{CSV_FILE}' já existe. Os novos dados serão adicionados.")

criar_cabecalho_csv()



SINAL_ATUAL = "NENHUM" 
GRAVANDO = False       # para saber se ta gravando
contador_amostras = 0  


cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        # deixa em rgb
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        
        # volta para BGR
        frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # desenha os pontos
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                
                
                
                if GRAVANDO:
                    keypoints = []
                    for landmark in hand_landmarks.landmark:
                        keypoints.extend([landmark.x, landmark.y, landmark.z])

                    # [Sinal, x0, y0, z0, x1, y1, z1, ...]
                    row = [SINAL_ATUAL] + keypoints
                    
                    with open(CSV_FILE, mode='a', newline='') as f: # 'a' é para ADICIONAR
                        csv_writer = csv.writer(f, delimiter=',')
                        csv_writer.writerow(row)
                    contador_amostras += 1

                
                    if contador_amostras >= LIMITE_AMOSTRAS:
                        GRAVANDO = False
                        print(SINAL_ATUAL)
                        SINAL_ATUAL = "NENHUM"
                        contador_amostras = 0 

        status_cor = (0, 0, 255) if GRAVANDO else (255, 255, 255)
        cv2.putText(frame, f"AMOSTRAS: {contador_amostras}/{LIMITE_AMOSTRAS}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"SINAL: {SINAL_ATUAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_cor, 2)
        cv2.putText(frame, f"GRAVANDO: {GRAVANDO}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, status_cor, 2)
        
        
        cv2.imshow('Coletor de Dados de Libras', frame)

        
        key = cv2.waitKey(10) & 0xFF


        # Pressione a letra desejada para começar a gravar.
        if not GRAVANDO:
            if key == ord('a'):
                SINAL_ATUAL = "A"
                GRAVANDO = True
                
            elif key == ord('b'):
                SINAL_ATUAL = "B"
                GRAVANDO = True
                
            elif key == ord('c'):
                SINAL_ATUAL = "C"
                GRAVANDO = True
 
            elif key == ord('d'):
                SINAL_ATUAL = "D"
                GRAVANDO = True

            elif key == ord('e'):
                SINAL_ATUAL = "E"
                GRAVANDO = True
  
            elif key == ord('f'):
                SINAL_ATUAL = "F"
                GRAVANDO = True
               
            elif key == ord('g'):
                SINAL_ATUAL = "G"
                GRAVANDO = True
  
            elif key == ord('h'):
                SINAL_ATUAL = "H"
                GRAVANDO = True
               
            elif key == ord('i'):
                SINAL_ATUAL = "I"
                GRAVANDO = True
       
            elif key == ord('j'):
                SINAL_ATUAL = "J"
                GRAVANDO = True

            elif key == ord('k'):
                SINAL_ATUAL = "K"
                GRAVANDO = True
     
            elif key == ord('l'):
                SINAL_ATUAL = "L"
                GRAVANDO = True
         
            elif key == ord('m'):
                SINAL_ATUAL = "M"
                GRAVANDO = True
     
            elif key == ord('n'):
                SINAL_ATUAL = "N"
                GRAVANDO = True
     
            elif key == ord('o'):
                SINAL_ATUAL = "O"
                GRAVANDO = True
  
            elif key == ord('p'):
                SINAL_ATUAL = "P"
                GRAVANDO = True
    
            elif key == ord('q'): 
                SINAL_ATUAL = "Q"
                GRAVANDO = True
    
            elif key == ord('r'):
                SINAL_ATUAL = "R"
                GRAVANDO = True
       
            elif key == ord('s'):
                SINAL_ATUAL = "S"
                GRAVANDO = True
          
            elif key == ord('t'):
                SINAL_ATUAL = "T"
                GRAVANDO = True
      
            elif key == ord('u'):
                SINAL_ATUAL = "U"
                GRAVANDO = True
    
            elif key == ord('v'):
                SINAL_ATUAL = "V"
                GRAVANDO = True

            elif key == ord('w'):
                SINAL_ATUAL = "W"
                GRAVANDO = True
         
            elif key == ord('x'):
                SINAL_ATUAL = "X"
                GRAVANDO = True
              
            elif key == ord('y'):
                SINAL_ATUAL = "Y"
                GRAVANDO = True
          
            elif key == ord('z'):
                SINAL_ATUAL = "Z"
                GRAVANDO = True
                
        

cap.release()
cv2.destroyAllWindows()