import dlib, cv2, os
import numpy as np
import matplotlib.pyplot as plt
from imutils import face_utils

weight_path = 'D:\Study/1.Project/(2)_Weight/mmod_human_face_detector.dat'

# 가중치 불러오기
detector = dlib.cnn_face_detection_model_v1(weight_path)

# image path
img_path = 'D:/image_kface_front'

filename = os.listdir(img_path)


for i, d in enumerate(filename):
    img = cv2.imread(img_path + '/' + d)
    dets = detector(img, upsample_num_times=1)

    for d in dets:
        x1, y1 = d.rect.left(), d.rect.top()
        x2, y2 = d.rect.right(), d.rect.bottom()
        pad = (x2 - x1)

        x1 = x1 - pad/2
        y1 = y1 - pad/2
        x2 = x2 + pad/2
        y2 = y2 + pad/2

        x1, x2, y1, y2 = map(int, (x1, x2, y1, y2))

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
    
        img = img[y1:y2, x1:x2]
        img = cv2.imwrite('D:/Study/1.Project/(7)_Image/kface/' + str(kface_cropped70000 + i) + '.jpg', img)