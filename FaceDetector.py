import cv2
import sys
import time
import numpy as np
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import os

from __int__ import MODEL_FILE


class FaceDetector:
    """面部检测器"""

    def __init__(self):
        # 使用dlib的人脸检测器和特征点预测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(MODEL_FILE)

        # 定义眼睛关键点索引
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # 状态变量
        self.face_rect = None
        self.landmarks = None
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_fps = 0

    def detect_faces(self, frame):
        """检测面部和特征点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        if len(rects) > 0:
            self.face_rect = rects[0]
            self.landmarks = self.predictor(gray, self.face_rect)
            self.landmarks = face_utils.shape_to_np(self.landmarks)
        else:
            self.face_rect = None
            self.landmarks = None

        # 计算FPS
        self.frame_count += 1
        if time.time() - self.start_time > 1:
            self.detection_fps = self.frame_count / (time.time() - self.start_time)
            self.frame_count = 0
            self.start_time = time.time()

        return self.face_rect is not None

    def get_face_roi(self, frame):
        """获取面部区域"""
        if self.face_rect is None:
            return None

        x, y, w, h = self.face_rect.left(), self.face_rect.top(), \
                     self.face_rect.width(), self.face_rect.height()
        return frame[y:y + h, x:x + w]

    def draw_face_info(self, frame):
        """绘制面部信息"""
        if self.face_rect is not None:
            # 绘制人脸框
            x, y, w, h = self.face_rect.left(), self.face_rect.top(), \
                         self.face_rect.width(), self.face_rect.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 显示FPS
        cv2.putText(frame, f"FPS: {self.detection_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame