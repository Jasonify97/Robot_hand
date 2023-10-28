import cv2 
import mediapipe as mp  
from PIL import ImageFont, ImageDraw, Image
import numpy as np  
import math    

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Input Webcam
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
      continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        
        # 폰트 설정
        font = ImageFont.truetype("fonts/Snow_crap.ttf", 20)
        
        # array형태로 이미지 띄우기
        image = Image.fromarray(image)
        
        # 이미지위에 텍스트/선을 쓰기 위해 선언
        draw = ImageDraw.Draw(image)
        
        # 각 관절의 사이각 함수
        # down : 목표 joint의 아래 관절 좌표
        # point : 목표 joint 관절 좌표
        # up : 목표 joint의 윗 관절 좌표
        def joint_degree(down, point, up): # PIP, DIP, TIP 관절
          below = np.array([down.x * 100,
                            down.y * 100,
                            down.z * 100])
          joint = np.array([point.x * 100,
                            point.y * 100,
                            point.z * 100])
          above = np.array([up.x * 100,
                            up.y * 100,
                            up.z * 100])
          # below - joint 벡터 계산
          below_joint = [joint[0] - below[0], joint[1] - below[1], joint[2] - below[2]]
          # joint - above 벡터 계산
          joint_above = [above[0] - joint[0], above[1] - joint[1], above[2] - joint[2]]

          # 내적 계산
          dot_product = below_joint[0] * joint_above[0] + below_joint[1] * joint_above[1] + below_joint[2] * joint_above[2]

          # ab 벡터와 bc 벡터의 길이 계산
          length_below_joint = math.sqrt(below_joint[0] ** 2 + below_joint[1] ** 2 + below_joint[2] ** 2)
          length_joint_above = math.sqrt(joint_above[0] ** 2 + joint_above[1] ** 2 + joint_above[2] ** 2)

          # 아크코사인을 이용하여 각을 라디안 단위로 계산
          angle_rad = math.acos(dot_product / (length_below_joint * length_joint_above))

          # 라디안을 도(degree)로 변환
          # 소수점 첫째자리까지 표현
          # PIL함수 특성상 float은 오류나기 때문에 str로 변환
          angle_deg = str(round(math.degrees(angle_rad),1)) + "도"
          
          # 각도 표시
          draw.text((point.x * image_width, point.y * image_height-20), angle_deg, font=font,fill=(255, 255, 255))
        
        def MCP_joint_degree(down, point, up): # MCP 관절
          below_xy = np.array([down.x * 100,
                              down.y * 100])
          below_yz = np.array([down.y * 100,
                              down.z * 100])
          
          joint_xy = np.array([point.x * 100,
                              point.y * 100])
          joint_yz = np.array([point.y * 100,
                              point.z * 100])
          
          above_xy = np.array([up.x * 100,
                            up.y * 100])
          above_yz = np.array([up.y * 100,
                            up.z * 100])
          
          m1_xy = (joint_xy[1] - below_xy[1]) / (joint_xy[0] - below_xy[0])
          m2_xy = (above_xy[1] - joint_xy[1]) / (above_xy[0] - joint_xy[0])

          m1_yz = (joint_yz[1] - below_yz[1]) / (joint_yz[0] - below_yz[0])
          m2_yz = (above_yz[1] - joint_yz[1]) / (above_yz[0] - joint_yz[0])
          
          # 두 직선 사이의 각 계산
          angle_xy = math.atan(abs((m2_xy - m1_xy) / (1 + m1_xy * m2_xy)))
          angle_yz = math.atan(abs((m2_yz - m1_yz) / (1 + m1_yz * m2_yz)))

          # 라디안을 도로 변환
          angle_deg_xy = str(round(math.degrees(angle_xy),1)) + "도"
          angle_deg_yz = str(round(math.degrees(angle_yz),1)) + "도"
          draw.text((point.x * image_width, point.y * image_height+10), angle_deg_xy, font=font,fill=(255, 255, 255))
          draw.text((point.x * image_width, point.y * image_height+30), angle_deg_yz, font=font,fill=(255, 255, 255))
        
        # 엄지
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]) # joint1
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]) # joint2
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]) # joint3
        
        # 검지
        MCP_joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]) # joint5
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]) # joint6
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]) # joint7
        
        # 중지
        MCP_joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]) # joint9
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]) # joint10
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) # joint11
        
        # 약지
        MCP_joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]) # joint9
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]) # joint10
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]) # joint11
        
        # 약지
        MCP_joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]) # joint9
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]) # joint10
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]) # joint11

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
# cap2.release()
cap.release()

cv2.destroyAllWindows()