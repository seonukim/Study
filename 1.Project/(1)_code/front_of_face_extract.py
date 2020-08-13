## 한국인 안면 이미지
## 정면 사진만 가져오기

import os
import cv2
import glob
import matplotlib.pyplot as plt

image_path = 'C:/Users/bitcamp/Downloads/Middle_Resolution/middle'
save_path = 'D:/image_kface_front'
# os.chdir(image_path)
list_dir = os.listdir(path=image_path)

for i in list_dir:
    img = image_path + '/' + i + '/S001/L1/E01/C7.jpg'
    img = cv2.imread(img)
    # img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)   # 리사이즈하면 이미지가 찌그러짐
    img = cv2.imwrite(save_path + '/' + i + '.jpg', img)



# plt.imshow(img)
# plt.show()