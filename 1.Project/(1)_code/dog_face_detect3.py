import dlib, cv2, os                            # os.walk뺀거
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

train_path = 'D:/backup/Downloads/project/breed'

def face_detector(path, folder, w, h):
    print('---------- START %s ---------'%(folder))
    image_dir = path + '/'+folder+'/'
    X = []

    f = os.listdir(image_dir)           # 폴더내 파일 이름 찾기

    for filename in f:                  # 파일 별로 이미지 불러오기
        img = cv2.imread(image_dir + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv는 BGR로 불러 들임으로 볼 때 우리가 원하는 색으로 보기 위해 RGB로

        img_result = img.copy()                     # 원본 이미지 copy

        detector = dlib.cnn_face_detection_model_v1('./weight/dogHeadDetector.dat')
        dets = detector(img, upsample_num_times=1)

        x = X.append

        for i, d in enumerate(dets):

            x1, y1 = d.rect.left(), d.rect.top()
            x2, y2 = d.rect.right(), d.rect.bottom()
            pad = (x2 - x1)

            x1 = x1 - pad/4
            y1 = y1 - pad*3/8
            x2 = x2 + pad/4
            y2 = y2 + pad/8

            x1, x2, y1, y2 = map(int, (x1, x2, y1, y2)) # int형으로 변환

            if x1 < 0:
                x1 = 0          
            if y1 < 0:
                y1 = 0
                    
            cropping = img[y1:y2, x1:x2]
            crop = cv2.resize(cropping, dsize = (w, h), interpolation = cv2.INTER_LINEAR)

            x(crop/255)
        
        ''' 견종별로 따로 따로 저장 '''
    images = np.array(X)

    np.save('./data/face_image_%s.npy'%(folder), images)

    print('---------- END %s ---------'%(folder))

    
face_detector(train_path, 'huskey', 300, 300)




