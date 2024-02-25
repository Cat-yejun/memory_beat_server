# -*- encoding: utf-8 -*-
# -------------------------------------------------#
# Date created          : 2020. 8. 18.
# Date last modified    : 2020. 8. 19.
# Author                : chamadams@gmail.com
# Site                  : http://wandlab.com
# License               : GNU General Public License(GPL) 2.0
# Version               : 0.1.0
# Python Version        : 3.6+
# -------------------------------------------------#

import time
import cv2
import numpy as np
from threading import Thread
from queue import Queue
import mediapipe as mp
import random


class MemoryBeat:

    def __init__(self):
        self.current_time = time.time()
        self.preview_time = time.time()
        print("initializing...")
        self.gesture = {
            0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
            6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'mem'
        }
        self.memory_gesture ={0: 'fist', 1: 'one', 11: 'mem'}

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 동그라미 그리기를 위한 변수
        self.first_circle_color = (122, 122, 122)  # 초록색
        self.first_circle_pos = (150, 280)  # 초기 위치
        self.second_circle_color = (122, 122, 122)
        self.second_circle_pos = (500, 280) 
        self.circle_radius = 50
        self.font_scale = 1
        self.text_size = cv2.getTextSize('Good!', cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2)[0]


        self.last_update_time = time.time()
        self.update_interval = 3  # 동그라미 위치 업데이트 간격 (초)

        self.first_color = 'gray'
        self.second_color = 'gray'
        
        self.left_hand_action = '?'
        self.right_hand_action = '?'
        
        # Gesture recognition model
        file = np.genfromtxt('data/memory_data.csv', delimiter=',')
        angle = file[:,:-1].astype(np.float32)
        label = file[:, -1].astype(np.float32)
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(angle, cv2.ml.ROW_SAMPLE, label)
        
        self.combo = 0
        self.rhythm = False
        self.prev_guess = False
        
        self.total_problems = 3
        self.current_problem = 0
        self.game_over = False

    @staticmethod
    def format_number(number):
        # 숫자를 세 자리 문자열로 포맷팅
        return f"{number:03d}"

    @staticmethod
    def load_digit_images(formatted_number, path_template):
        # 포맷된 숫자에 대한 각 자리수의 이미지를 불러옴
        digit_images = [cv2.imread(path_template.format(digit), cv2.IMREAD_UNCHANGED) for digit in formatted_number]
        return digit_images

    @staticmethod
    def combine_images_horizontally(images):
        # 이미지들을 가로로 결합
        return cv2.hconcat(images)

    @staticmethod
    def overlay_image(background, overlay, position="top_center", scale=0.5):
        # 배경 이미지의 크기와 중앙 좌표를 계산
        bg_h, bg_w = background.shape[:2]
        bg_center_x = bg_w // 2
        bg_center_y = bg_h // 2

        # 오버레이 이미지의 크기를 조절
        ovr_h, ovr_w = overlay.shape[:2]
        new_h, new_w = int(ovr_h * scale), int(ovr_w * scale)
        overlay_resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # # 오버레이할 위치 계산
        # if position == "top_center":
        #     x = bg_center_x - (new_w // 2)
        #     y = int(bg_h * 0.2)  # 화면 상단에 약간의 여백을 두고 배치
        # elif position == "center":
        #     x = bg_center_x - (new_w // 2)
        #     y = bg_center_y - (new_h)
        # elif position == "bottom_center":
        #     x = bg_center_x - (new_w // 2)
        #     y = int(bg_h * 0.5)
        # else:
        #     x, y = 0, 0  # 기본 위치
        
        # 오버레이할 위치 계산
        if position == "top_center":
            x = bg_center_x - (new_w // 2)
            y = bg_center_y - (new_h // 2) - bg_h // 8 - bg_h // 18  # 화면 상단에 약간의 여백을 두고 배치
        elif position == "center":
            x = bg_center_x - (new_w // 2)
            y = bg_center_y - (new_h // 2) - bg_h // 18
        elif position == "bottom_center":
            x = bg_center_x - (new_w // 2)
            y = bg_center_y - (new_h // 2) + bg_h // 8 - bg_h // 18
        else:
            x, y = 0, 0  # 기본 위치
        
        # 배경 이미지의 오버레이할 부분 추출
        overlay_area = background[y:y+new_h, x:x+new_w]

        # 알파 채널을 이용한 오버레이
        alpha = overlay_resized[:, :, 3] / 255.0
        for c in range(0, 3):
            overlay_area[:, :, c] = (1. - alpha) * overlay_area[:, :, c] + alpha * overlay_resized[:, :, c]

        # 오버레이된 부분을 다시 배경 이미지에 덮어쓰기
        background[y:y+new_h, x:x+new_w] = overlay_area

    def show_combo(self, frame, combo):
        formatted_number = self.format_number(combo)
        digit_images = self.load_digit_images(formatted_number, "numbers/{}.png")  # 이미지 경로 템플릿에 맞게 수정 필요
        combined_image = self.combine_images_horizontally(digit_images)
        self.overlay_image(frame, combined_image, position="top_center", scale=0.08)  # 스케일 조정 가능

    def show_result(self, frame, result):
        formatted_number = self.format_number(result)
        digit_images = self.load_digit_images(formatted_number, "numbers/{}.png")
        combined_image = self.combine_images_horizontally(digit_images)
        self.overlay_image(frame, combined_image, position="center", scale=0.08)
        
        answer1 = cv2.imread("numbers/answer.png", cv2.IMREAD_UNCHANGED)
        self.overlay_image(frame, answer1, position="top_center", scale=0.3)
        
        answer2 = cv2.imread("numbers/answer2.png", cv2.IMREAD_UNCHANGED)
        self.overlay_image(frame, answer2, position="bottom_center", scale=0.25)

    def process_frame(self, frame, width, height):
        frame.flags.writeable = False
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.first_circle_pos = (int(width * 0.35), int(height * 0.7))
        self.second_circle_pos = (int(width * 0.65), int(height * 0.7))
        self.radius = int(min(width, height) / 12)
        self.font_scale = 2 * min(width, height) / 1000
        
        if time.time() - self.last_update_time > self.update_interval:
            # 동그라미 색상 및 위치 무작위 업데이트
            self.first_color = random.choice(['red', 'green', 'blue'])
            self.second_color = random.choice(['red', 'green', 'blue'])
            
            if self.first_color == 'red':
                self.first_circle_color = (255, 0, 0)
            elif self.first_color == 'green':
                self.first_circle_color = (0, 255, 0)
            else:
                self.first_circle_color = (0, 0, 255)
                
            if self.second_color == 'red':
                self.second_circle_color = (255, 0, 0)
            elif self.second_color == 'green':
                self.second_circle_color = (0, 255, 0)
            else:
                self.second_circle_color = (0, 0, 255)
                
            self.last_update_time = time.time()
            self.rhythm = False
            self.current_problem += 1
        
            # if(self.prev_guess == False):
            #     self.combo = 0
            # else:
            #     self.combo += 1
            if(self.prev_guess == True):
                self.combo += 1
        
        cv2.circle(frame, self.first_circle_pos, self.circle_radius, self.first_circle_color, -1)
        cv2.circle(frame, self.second_circle_pos, self.circle_radius, self.second_circle_color, -1)
        
        results = self.hands.process(frame)
        
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if(self.current_problem > self.total_problems):
                self.game_over = True
                return frame

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_side = 'left' if handedness.classification[0].label == 'Left' else 'right'  # 미디어파이프는 카메라 반전을 고려

                # 각 손의 랜드마크를 기반으로 조인트 각도 계산
                joint = np.zeros((21, 3))
                for j, lm in enumerate(hand_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                angle = np.degrees(angle)

                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = self.knn.findNearest(data, 3)
                idx = int(results[0][0])

                if idx in self.memory_gesture:
                    action = self.memory_gesture[idx]
                    if hand_side == 'left':
                        self.left_hand_action = action
                        org = (int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0]))
                        cv2.putText(frame, text=self.memory_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=self.font_scale, color=(255, 255, 255), thickness=2)

                    else:
                        self.right_hand_action = action
                        org = (int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0]))
                        cv2.putText(frame, text=self.memory_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=self.font_scale, color=(255, 255, 255), thickness=2)


                # self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # 여기서는 예시로 왼손과 오른손의 동작을 각각 판별한 뒤, 조건을 만족하는지 확인
        # 실제 구현에서는 화면에 표시되는 동그라미의 색상과 위치에 따라 조건을 설정해야 함

        first_correct_action =  (self.first_color == 'red' and self.left_hand_action == 'fist') or \
                                (self.first_color == 'green' and self.left_hand_action == 'one') or \
                                (self.first_color == 'blue' and self.left_hand_action == 'mem')
                                
        second_correct_action = (self.second_color == 'red' and self.right_hand_action == 'fist') or \
                                (self.second_color == 'green' and self.right_hand_action == 'one') or \
                                (self.second_color == 'blue' and self.right_hand_action == 'mem')
                                    

            # Add any additional processing (e.g., gesture recognition) here
        print(first_correct_action, second_correct_action, self.first_color, self.second_color, self.left_hand_action, self.right_hand_action)
    
        if first_correct_action and second_correct_action:
            position = (int(0.5 * frame.shape[1]), int(0.7 * frame.shape[0]))
        
            # 시작 위치를 텍스트 크기를 기반으로 조정하여 중앙 정렬
            text_x = position[0] - self.text_size[0] // 2
            text_y = position[1] + self.text_size[1] // 2
            
            # 이미지에 텍스트를 추가
            cv2.putText(frame, 'Good!', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255,255,255), 2)
            if(self.rhythm == False):
                self.rhythm = True
            self.prev_guess = True
        else:
            self.prev_guess = False
        
        # if(self.combo > 0):
        self.show_combo(frame, self.combo)

        return frame
    
    def gameOver(self):
        return self.game_over
    
    def gameOverScreen(self, frame, width, height):
        frame.flags.writeable = False
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self.show_result(frame, self.combo)
        
        return frame
        
    
   