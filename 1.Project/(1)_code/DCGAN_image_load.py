import h5py
import cv2
import os
import numpy as np

def load_image(path):
    f = os.listdir(path)

    X = []

    x = X.append
    for filename in f:
        img = cv2.imread(path +'/'+ filename)
        img_umat = cv2.UMat(img)
        img_resize = cv2.resize(img_umat, (128, 128), interpolation = cv2.INTER_LINEAR)
        img_result = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        img_result = cv2.UMat.get(img_result)

        x(img_result)

    return np.array(X)