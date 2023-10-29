import cv2 
import mediapipe as mp  
from PIL import ImageFont, ImageDraw, Image
import numpy as np  
import math    

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Input 2 Webcam
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)

with mp_hands.Hands(
    # 인식할 손 개수 
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cap1.isOpened() :
    success_cap1, image_cap1 = cap1.read()
    success_cap2, image_cap2 = cap2.read()
    
    if not success_cap1:
      print("Alert cap1")
      continue
    
    if not success_cap2:
      print("Alert cap2")
      continue
    
    # 이미지 filp(좌우반전), cv2형식에 맞게 RGB로 바꾸기
    image_cap1 = cv2.cvtColor(cv2.flip(image_cap1, 1), cv2.COLOR_BGR2RGB)
    image_cap2 = cv2.cvtColor(cv2.flip(image_cap2, 1), cv2.COLOR_BGR2RGB)
    
    # 이미지 읽기 전용으로 바꾸기
    image_cap1.flags.writeable = False
    image_cap2.flags.writeable = False
    # hands 값 읽어오기
    results_1 = hands.process(image_cap1)
    results_2 = hands.process(image_cap2)
    
    # 이미지 수정 전용으로 바꾸기
    image_cap1.flags.writeable = True
    image_cap2.flags.writeable = True
    
    # 이미지 GBR로 바꾸기
    image_cap1 = cv2.cvtColor(image_cap1, cv2.COLOR_RGB2BGR)
    image_cap2 = cv2.cvtColor(image_cap2, cv2.COLOR_RGB2BGR)
    
    # 이미지 크기 읽어오기
    image_height_cap1, image_width_cap1, _ = image_cap1.shape
    image_height_cap2, image_width_cap2, _ = image_cap2.shape
    
    # 폰트 설정
    font = ImageFont.truetype("fonts/Snow_crap.ttf", 20)
    
    if results_1.multi_hand_landmarks and results_2.multi_hand_landmarks:
      for hand_landmarks, hand_landmarks1 in zip(results_1.multi_hand_landmarks, results_2.multi_hand_landmarks):

        # array형태로 이미지 띄우기
        image_cap1 = Image.fromarray(image_cap1)
        
        # 이미지위에 텍스트/선을 쓰기 위해 선언
        draw = ImageDraw.Draw(image_cap1)
      
        # down : 목표 joint의 아래 관절 좌표, point : 목표 joint 관절 좌표, up : 목표 joint의 윗 관절 좌표
        def joint_degree(down, point, up, down1, point1, up1): # PIP, DIP, TIP 관절
          below = np.array([down.x * 100,
                            down.y * 100,
                            down.z * 100])
          joint = np.array([point.x * 100,
                            point.y * 100,
                            point.z * 100])
          above = np.array([up.x * 100,
                            up.y * 100,
                            up.z * 100])
          below1 = np.array([down1.x * 100,
                            down1.y * 100,
                            down1.z * 100])
          joint1 = np.array([point1.x * 100,
                            point1.y * 100,
                            point1.z * 100])
          above1 = np.array([up1.x * 100,
                            up1.y * 100,
                            up1.z * 100])
          # below - joint 벡터 계산
          below_joint = [joint[0] - below[0], joint[1] - below[1], joint[2] - below[2]]
          below_joint1 = [joint1[0] - below1[0], joint1[1] - below1[1], joint1[2] - below1[2]]
          # joint - above 벡터 계산
          joint_above = [above[0] - joint[0], above[1] - joint[1], above[2] - joint[2]]
          joint_above1 = [above1[0] - joint1[0], above1[1] - joint1[1], above1[2] - joint1[2]]

          # 내적 계산
          dot_product = below_joint[0] * joint_above[0] + below_joint[1] * joint_above[1] + below_joint[2] * joint_above[2]
          dot_product1 = below_joint1[0] * joint_above1[0] + below_joint1[1] * joint_above1[1] + below_joint1[2] * joint_above1[2]
          # ab 벡터와 bc 벡터의 길이 계산
          length_below_joint = math.sqrt(below_joint[0] ** 2 + below_joint[1] ** 2 + below_joint[2] ** 2)
          length_joint_above = math.sqrt(joint_above[0] ** 2 + joint_above[1] ** 2 + joint_above[2] ** 2)
          
          length_below_joint1 = math.sqrt(below_joint1[0] ** 2 + below_joint1[1] ** 2 + below_joint1[2] ** 2)
          length_joint_above1 = math.sqrt(joint_above1[0] ** 2 + joint_above1[1] ** 2 + joint_above1[2] ** 2)

          # 아크코사인을 이용하여 각을 라디안 단위로 계산
          angle_rad = math.acos(dot_product / (length_below_joint * length_joint_above))
          angle_rad1 = math.acos(dot_product1 / (length_below_joint1 * length_joint_above1))
          # 두개 각도 평균값
          degree = (angle_rad+angle_rad1)/2

          # 라디안을 도(degree)로 변환
          # 소수점 첫째자리까지 표현
          # PIL함수 특성상 float은 오류나기 때문에 str로 변환
          angle_deg = str(round(math.degrees(degree),1)) + "도"
          
          # 각도 표시
          draw.text((point.x * image_width_cap1, point.y * image_height_cap1-20), angle_deg, font=font,fill=(255, 255, 255))
        def MCP_joint_degree(down, point, up, down1, point1, up1): # MCP 관절
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
          
          below_xy1 = np.array([down1.x * 100,
                              down1.y * 100])
          below_yz1 = np.array([down1.y * 100,
                              down1.z * 100])
          joint_xy1 = np.array([point1.x * 100,
                              point1.y * 100])
          joint_yz1 = np.array([point1.y * 100,
                              point1.z * 100])
          above_xy1 = np.array([up1.x * 100,
                            up1.y * 100])
          above_yz1 = np.array([up1.y * 100,
                            up1.z * 100])
          
          m1_xy = (joint_xy[1] - below_xy[1]) / (joint_xy[0] - below_xy[0])
          m2_xy = (above_xy[1] - joint_xy[1]) / (above_xy[0] - joint_xy[0])

          m1_yz = (joint_yz[1] - below_yz[1]) / (joint_yz[0] - below_yz[0])
          m2_yz = (above_yz[1] - joint_yz[1]) / (above_yz[0] - joint_yz[0])
          
          m1_xy1 = (joint_xy1[1] - below_xy1[1]) / (joint_xy1[0] - below_xy1[0])
          m2_xy1 = (above_xy1[1] - joint_xy1[1]) / (above_xy1[0] - joint_xy1[0])

          m1_yz1 = (joint_yz1[1] - below_yz1[1]) / (joint_yz1[0] - below_yz1[0])
          m2_yz1 = (above_yz1[1] - joint_yz1[1]) / (above_yz1[0] - joint_yz1[0])
          
          # 두 직선 사이의 각 계산
          angle_xy = math.atan(abs((m2_xy - m1_xy) / (1 + m1_xy * m2_xy)))
          angle_yz = math.atan(abs((m2_yz - m1_yz) / (1 + m1_yz * m2_yz)))
          
          angle_xy1 = math.atan(abs((m2_xy1 - m1_xy1) / (1 + m1_xy1 * m2_xy1)))
          angle_yz1 = math.atan(abs((m2_yz1 - m1_yz1) / (1 + m1_yz1 * m2_yz1)))
          
          degree = (angle_xy+angle_xy1)/2
          
          # 라디안을 도로 변환
          angle_deg_xy = str(round(math.degrees(degree),1)) + "도"

          draw.text((point.x * image_width_cap1, point.y * image_height_cap1+10), angle_deg_xy, font=font,fill=(255, 255, 255))

        def thumb_cmc(point, point1, point2, point3, point_1, point1_1, point2_1, point3_1):# THUMB_CMC 관절
          ###### 첫번째 cam
          # 1번째 평면 벡터 계산
          space1_vector1 = np.array([point2.x * 100 - point1.x * 100, point2.y * 100 - point1.y * 100, point2.z * 100 - point1.z * 100])
          space1_vector2 = np.array([point3.x * 100 - point1.x * 100, point3.y * 100 - point1.y * 100, point3.z * 100 - point1.z * 100])
          # 2번째 평면 벡터 계산
          space2_vector1 = np.array([point2.x * 100 - point1.x * 100, point2.y * 100 - point1.y * 100, point2.z * 100 - point1.z * 100])
          space2_vector2 = np.array([point3.x * 100 - point1.x * 100, point3.y * 100 - point1.y * 100, point3.z * 100 - point1.z * 100])

          # 1번째 평면 법선 벡터
          normal_vector_space1 = np.cross(space1_vector1, space1_vector2)
          # 2번째 평면 법선 벡터
          normal_vector_space2 = np.cross(space2_vector1, space2_vector2)
          
          dot_product = normal_vector_space1[0] * normal_vector_space2[0] + normal_vector_space1[1] * normal_vector_space2[1] + normal_vector_space1[2] * normal_vector_space2[2]
          
          # 법선 벡터의 길이 계산
          magnitude1 = np.sqrt(normal_vector_space1[0]**2 + normal_vector_space1[1]**2 + normal_vector_space1[2]**2)
          magnitude2 = np.sqrt(normal_vector_space2[0]**2 + normal_vector_space2[1]**2 + normal_vector_space2[2]**2)

          # 각도 계산
          cos_theta = dot_product / (magnitude1 * magnitude2)
          theta = np.arccos(cos_theta)
          
          ###### 두번째 cam
          # 1번째 평면 벡터 계산
          space1_vector1 = np.array([point2.x * 100 - point1_1.x * 100, point2_1.y * 100 - point1_1.y * 100, point2_1.z * 100 - point1_1.z * 100])
          space1_vector2 = np.array([point3.x * 100 - point1_1.x * 100, point3_1.y * 100 - point1_1.y * 100, point3_1.z * 100 - point1_1.z * 100])
          # 2번째 평면 벡터 계산
          space2_vector1 = np.array([point2.x * 100 - point1_1.x * 100, point2_1.y * 100 - point1_1.y * 100, point2_1.z * 100 - point1_1.z * 100])
          space2_vector2 = np.array([point3.x * 100 - point1_1.x * 100, point3_1.y * 100 - point1_1.y * 100, point3_1.z * 100 - point1_1.z * 100])

          # 1번째 평면 법선 벡터
          normal_vector_space1 = np.cross(space1_vector1, space1_vector2)
          # 2번째 평면 법선 벡터
          normal_vector_space2 = np.cross(space2_vector1, space2_vector2)
          
          dot_product = normal_vector_space1[0] * normal_vector_space2[0] + normal_vector_space1[1] * normal_vector_space2[1] + normal_vector_space1[2] * normal_vector_space2[2]
          
          # 법선 벡터의 길이 계산
          magnitude1 = np.sqrt(normal_vector_space1[0]**2 + normal_vector_space1[1]**2 + normal_vector_space1[2]**2)
          magnitude2 = np.sqrt(normal_vector_space2[0]**2 + normal_vector_space2[1]**2 + normal_vector_space2[2]**2)

          # 각도 계산
          cos_theta = dot_product / (magnitude1 * magnitude2)
          theta_1 = np.arccos(cos_theta)

          
          # 라디안에서 도로 변환
          degree = str(round(math.degrees((theta+theta_1)/2),1)) + "도"
          
          draw.text((point.x * image_width_cap1, point.y * image_height_cap1+10), degree, font=font,fill=(255, 255, 255))
        # 엄지
        thumb_cmc(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC],
                  hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                  hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                  hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                  hand_landmarks1.landmark[mp_hands.HandLandmark.THUMB_CMC],
                  hand_landmarks1.landmark[mp_hands.HandLandmark.PINKY_MCP],
                  hand_landmarks1.landmark[mp_hands.HandLandmark.WRIST],
                  hand_landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]) # joint1
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.THUMB_CMC],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.THUMB_MCP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.THUMB_IP]) # joint2
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.THUMB_MCP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.THUMB_IP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.THUMB_TIP]) # joint3
        
        # 검지
        MCP_joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                        hand_landmarks1.landmark[mp_hands.HandLandmark.WRIST],
                        hand_landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                        hand_landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]) # joint5
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]) # joint6
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]) # joint7
        
        # 중지
        MCP_joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                        hand_landmarks1.landmark[mp_hands.HandLandmark.WRIST],
                        hand_landmarks1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                        hand_landmarks1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]) # joint9
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]) # joint10
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) # joint11
        
        # 약지
        MCP_joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
                        hand_landmarks1.landmark[mp_hands.HandLandmark.WRIST],
                        hand_landmarks1.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
                        hand_landmarks1.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]) # joint9
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]) # joint10
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.RING_FINGER_DIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]) # joint11
        
        # 새끼
        MCP_joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP],
                        hand_landmarks1.landmark[mp_hands.HandLandmark.WRIST],
                        hand_landmarks1.landmark[mp_hands.HandLandmark.PINKY_MCP],
                        hand_landmarks1.landmark[mp_hands.HandLandmark.PINKY_PIP]) # joint9
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.PINKY_MCP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.PINKY_PIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.PINKY_DIP]) # joint10
        
        joint_degree(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.PINKY_PIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.PINKY_DIP],
                    hand_landmarks1.landmark[mp_hands.HandLandmark.PINKY_TIP]) # joint11

        image_cap1 = np.array(image_cap1)
        image_cap2 = np.array(image_cap2)
        # 손가락 뼈대 그리기
        mp_drawing.draw_landmarks(
            image_cap1,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())   


    cv2.imshow('Camera_Mac', image_cap1)
    cv2.imshow('Camera_phone', image_cap2)

    # esc 클릭시 종료
    if cv2.waitKey(5) & 0xFF == 27:
      break
    
cap1.release()
cap2.release()

cv2.destroyAllWindows()