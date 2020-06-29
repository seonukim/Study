import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import warnings ; warnings.filterwarnings('ignore')

def aidemy_imshow(name, img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()

cv2.imshow = aidemy_imshow