import dlib, cv2, os
import numpy as np
import matplotlib.pyplot as plt
from imutils import face_utils

weight_path = './(2)_Weight/mmod_human_face_detector.dat'
land_path = ''

# 가중치 불러오기
detector = dlib.cnn_face_detection_model_v1(weight_path)
predictor = dlib.shape_predictor()