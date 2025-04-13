import cv2
import sys
import time
import numpy as np
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import os

from DriverMonitor import DriverMonitor
from FaceDetector import FaceDetector


class DriverAssistanceSystem:
    """驾驶辅助系统主类"""

    def __init__(self):
        # 初始化视频捕获
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")

        # 初始化各模块
        self.face_detector = FaceDetector()
        self.driver_monitor = DriverMonitor(self.face_detector)

        # 状态变量
        self.is_running = False
        self.show_face_detection = True

    def run(self):
        """运行主循环"""
        self.is_running = True

        try:
            while self.is_running:
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print("Warning: Unable to read video frame")
                    break

                # 检测面部
                face_detected = self.face_detector.detect_faces(frame)

                # 初始化状态变量
                status_text = ""
                ear = 0.0

                # 如果检测到面部，分析驾驶员状态
                if face_detected and self.face_detector.landmarks is not None:
                    analyzed_face, status_text, ear = self.driver_monitor.analyze(
                        frame, self.face_detector.landmarks)
                else:

                    status_text = "No Face Detected"
                    cv2.putText(frame, status_text,
                                (frame.shape[1] // 2 - 100, frame.shape[0] - 30),  # 底部居中位置
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 红色文本

                # 显示面部检测框
                if self.show_face_detection and face_detected:
                    frame = self.face_detector.draw_face_info(frame)

                # 显示状态信息
                if face_detected:
                    y_offset = 60
                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 30
                    cv2.putText(frame, f"PERCLOS: {self.driver_monitor.perclos:.2f}",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 30
                    cv2.putText(frame, f"Status: {status_text}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 显示控制提示
                cv2.putText(frame, "ESC:Quit  F:Toggle Detection",
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 显示帧
                cv2.imshow("Driver Assistance System", frame)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键退出
                    break
                elif key == ord('f'):
                    self.show_face_detection = not self.show_face_detection

        except KeyboardInterrupt:
            print("系统被用户中断")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        self.is_running = False
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()