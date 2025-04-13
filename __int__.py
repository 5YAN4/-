import cv2
import sys
import time
import numpy as np
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import os


# 初始化设置
# 初始化pygame音频
pygame.mixer.init()

# 模型文件路径
MODEL_FILE = "shape_predictor_68_face_landmarks.dat"

# 检查模型文件是否存在
if not os.path.exists(MODEL_FILE):
    print(f"错误: 请下载并放置人脸关键点模型文件 '{MODEL_FILE}' 到当前目录")
    print("下载地址: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    sys.exit(1)
