import cv2 
import mediapipe as mp  
from PIL import ImageFont, ImageDraw, Image
import numpy as np  
import math    

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# For webcam input:
cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    # 인식할 손 개수 
    max_num_hands=1,
    
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

 
  while cap.isOpened():
    success, image = cap.read()

    if not success:
      print("Alert")

      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)


    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        
        # 폰트 설정
        font = ImageFont.truetype("fonts/Snow_crap.ttf", 40)
        
        # array형태로 이미지 띄우기
        image = Image.fromarray(image)
        
        # 이미지위에 텍스트/선을 쓰기 위해 선언
        draw = ImageDraw.Draw(image)
        #5번위치 좌표값
        INDEX_FINGER_MCP_5 = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * 100,
                                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * 100,
                                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z * 100])
        #6번위치 좌표값
        INDEX_FINGER_MCP_6 = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * 100,
                                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * 100,
                                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z * 100])
        #7번위치 좌표값
        INDEX_FINGER_MCP_7 = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * 100,
                                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * 100,
                                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z * 100])
        #8번위치 좌표값
        # INDEX_FINGER_MCP_8 = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
        #                                 hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
        #                                 hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z])
        
        # ab 벡터 계산
        ab = [INDEX_FINGER_MCP_6[0] - INDEX_FINGER_MCP_5[0], INDEX_FINGER_MCP_6[1] - INDEX_FINGER_MCP_5[1], INDEX_FINGER_MCP_6[2] - INDEX_FINGER_MCP_5[2]]

        # bc 벡터 계산
        bc = [INDEX_FINGER_MCP_7[0] - INDEX_FINGER_MCP_6[0], INDEX_FINGER_MCP_7[1] - INDEX_FINGER_MCP_6[1], INDEX_FINGER_MCP_7[2] - INDEX_FINGER_MCP_6[2]]

        # 내적 계산
        dot_product = ab[0] * bc[0] + ab[1] * bc[1] + ab[2] * bc[2]

        # ab 벡터와 bc 벡터의 길이 계산
        length_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
        length_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

        # 아크코사인을 이용하여 각을 라디안 단위로 계산
        angle_rad = math.acos(dot_product / (length_ab * length_bc))

        # 라디안을 도(degree)로 변환
        angle_deg = str(math.degrees(angle_rad))
        
        
        w, h = 400, 60

        x = 50
        y = 50

        # draw.rounded_rectangle((x, y, x + w, y + h), fill='black')
        draw.text((x, y), angle_deg, font=font, fill=(255, 255, 255))

        image = np.array(image)

        # 손가락 뼈대 그리기
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow('MediaPipe Hands', image)

    # esc 클릭시 종료
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()