import cv2
import sys
import time
import numpy as np
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import os


class DriverMonitor:
    """驾驶员状态监测"""

    def __init__(self, face_detector):
        # 从face_detector获取眼睛关键点索引
        self.lStart = face_detector.lStart
        self.lEnd = face_detector.lEnd
        self.rStart = face_detector.rStart
        self.rEnd = face_detector.rEnd

        # 眼睛状态参数
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 5
        self.PERCLOS_THRESH = 0.35

        # 状态变量
        self.eye_closed = False
        self.eye_counter = 0
        self.perclos = 0
        self.ear_history = []
        self.alert_level = 0  # 0=正常, 1=轻度警告, 2=严重警告
        self.is_distracted = False

        # 初始化警报系统
        try:
            self.alert_sound = pygame.mixer.Sound("alarm.wav")
        except:
            print("警告: 无法加载警报声音文件，使用系统蜂鸣声")
            self.alert_sound = None

    def eye_aspect_ratio(self, eye):
        """计算眼睛纵横比"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def update_perclos(self, ear):
        """更新PERCLOS值"""
        self.ear_history.append(ear < self.EYE_AR_THRESH)
        if len(self.ear_history) > 50:  # 使用最近50帧计算
            self.ear_history.pop(0)

        if len(self.ear_history) == 50:
            self.perclos = sum(self.ear_history) / len(self.ear_history)

    def check_distraction(self, landmarks, frame_shape):
        """检查是否分心"""
        if landmarks is None:
            return False

        # 检查面部中心是否偏离图像中心
        face_center = np.mean(landmarks, axis=0)
        img_center = (frame_shape[1] // 2, frame_shape[0] // 2)
        offset = np.linalg.norm(face_center - img_center)

        return offset > frame_shape[1] * 0.15

    def analyze(self, frame, landmarks):
        """分析驾驶员状态"""
        if landmarks is None:
            return frame, "No Face Detected", 0.8

        # 获取眼睛区域
        left_eye = landmarks[self.lStart:self.lEnd]
        right_eye = landmarks[self.rStart:self.rEnd]

        # 计算眼睛纵横比
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # 更新PERCLOS
        self.update_perclos(ear)

        # 检查眼睛状态
        if ear < self.EYE_AR_THRESH:
            self.eye_counter += 1
            if self.eye_counter >= self.EYE_AR_CONSEC_FRAMES:
                self.eye_closed = True
        else:
            self.eye_counter = 0
            self.eye_closed = False

        # 检查分心状态
        self.is_distracted = self.check_distraction(landmarks, frame.shape)

        # 更新警报级别
        if self.eye_closed and self.perclos > self.PERCLOS_THRESH:
            self.alert_level = 2
            self.trigger_alert()
        elif self.eye_closed or self.perclos > self.PERCLOS_THRESH or self.is_distracted:
            self.alert_level = 1
        else:
            self.alert_level = 0

        # 绘制结果
        result = self.draw_results(frame, left_eye, right_eye, ear)
        status = self.get_status_text()

        return result, status, ear

    def trigger_alert(self):
        """触发警报"""
        if self.alert_sound:
            if not pygame.mixer.get_busy():
                self.alert_sound.play()
        else:
            print('\a', end='', flush=True)

    def get_status_text(self):
        """获取状态文本"""
        status = []
        if self.eye_closed or self.perclos > self.PERCLOS_THRESH:
            status.append("Fatigue Driving")
        if self.is_distracted:
            status.append("Distracted")
        return "Normal" if not status else " ".join(status)

    def draw_results(self, frame, left_eye, right_eye, ear):
        """绘制分析结果"""
        # 绘制眼睛轮廓
        if left_eye is not None and right_eye is not None:
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        # 获取画面尺寸
        height, width = frame.shape[:2]

        # 显示信息
        if left_eye is None or right_eye is None:
            text = "No Face Detected"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 20  # 底部留20像素边距
            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # 检测到人脸时按原位置显示信息
            y_offset = 60
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
            cv2.putText(frame, f"PERCLOS: {self.perclos:.2f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

            # 显示状态
            color = (0, 255, 0) if self.alert_level == 0 else \
                (0, 165, 255) if self.alert_level == 1 else \
                    (0, 0, 255)
            status_text = self.get_status_text()
            cv2.putText(frame, f"Status: {status_text}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame