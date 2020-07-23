import numpy as np
import cv2, os, dlib
from class_project import Project
import time

start = time.time()

train_path = 'D:/data/project/testset/'

''' 만들어둔 Project class 사용 '''

def image_label(path, w, h):
    p = Project()

    X = []
    Y = []

    f = os.listdir(path)

    for filename in f:
        img = cv2.imread(path+filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        det_path = './project/project02/weight/dogHeadDetector.dat'
        dets = p.face_detector(img, det_path)

        filename = filename.replace('.jpg','').replace('.png','')
        print(filename)

        x = X.append
        y = Y.append

        for i, d in enumerate(dets):
            bbox = p.BBox(i, d)
            
            img_crop = p.crop()
            img_result = cv2.resize(img_crop, dsize = (w, h), interpolation= cv2.INTER_LINEAR)
                    
            x(img_result/255)
            y(filename)

    with open('./project/project02/data/pred_image_name.txt', 'w') as f:
        for line in Y:
            f.write(line+'\n')

    return np.array(X), np.array(Y)

x, y= image_label(train_path, 128, 128)
print(x.shape)

np.save('./project/project02/data/pred_image03.npy', x)
print('------------- save complete ---------------')

end = time.time()
print('Time :', end - start)