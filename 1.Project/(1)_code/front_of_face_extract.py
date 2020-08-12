import os
import cv2
import glob

image_path = 'C:/Users/bitcamp/Downloads/Middle_Resolution/middle'
# os.chdir(image_path)
list_dir = os.listdir(path=image_path)

for i in list_dir:
    img = image_path + '/' + i + '/S001/L1/E01/C7.jpg'
    img = cv2.imread(img)