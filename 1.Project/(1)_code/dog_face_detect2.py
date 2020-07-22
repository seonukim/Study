import dlib, cv2, os                            # os.walk뺀거
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

train_path = 'D:/backup/Downloads/project/breed'

def face_detector(path, w, h):
    folder_name = os.listdir(path)               # category 폴더 : list
    num_classes = len(folder_name)
    
    for idex, folder in enumerate(folder_name):                   # 폴더별 이미지 불러오기
        print('---------- START %s ---------'%(folder))
        # label = [0 for i in range(num_classes)]                   # one_hot_encoding
        # label[idex] = 1
        label = idex                                                # 단순 category 숫자
        image_dir = path + '/'+folder+'/'

        X = []
        Y = []

        f = os.listdir(image_dir)           # 폴더내 파일 이름 찾기

        for filename in f:                  # 파일 별로 이미지 불러오기
            img = cv2.imread(image_dir + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv는 BGR로 불러 들임으로 볼 때 우리가 원하는 색으로 보기 위해 RGB로

            img_result = img.copy()                     # 원본 이미지 copy

            detector = dlib.cnn_face_detection_model_v1('./weight/dogHeadDetector.dat')
            dets = detector(img, upsample_num_times=1)


            x = X.append
            y = Y.append

            for i, d in enumerate(dets):
                # print("Detection {}: Left:{} Top:{} Right:{} Bottom:{} Confidence:{}".format(i+1, 
                #             d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

                x1, y1 = d.rect.left(), d.rect.top()
                x2, y2 = d.rect.right(), d.rect.bottom()
                pad = (x2 - x1)
                    
                #---------------- bbox 키우기 ----------------
                x1 = x1 - pad/4
                y1 = y1 - pad*3/8
                x2 = x2 + pad/4
                y2 = y2 + pad/8
                #-------------정사각형으로 만들기--------------
                # dx = (x2 - x1)
                # dy = (y2 - y1)
                # same = np.abs(dx - dy)/2

                # if dx > dy:
                #     y1 = y1 - same
                #     y2 = y2 + same
                # else:
                #     x1 = x1 - same
                #     x2 = x2 + same
                #---------------------------------------------

                x1, x2, y1, y2 = map(int, (x1, x2, y1, y2)) # int형으로 변환

                # print('가로 : ',x1, x2)
                # print('세로 : ',y1, y2)                    

                # cv2.rectangle(img_result, (x1, y1), (x2, y2), thickness=1, color=(122, 122, 122), lineType=cv2.LINE_AA)

                # plt.figure(figsize=(16, 16))
                # plt.imshow(img_result)
                # plt.show()

                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                    
                cropping = img[y1:y2, x1:x2]
                crop = cv2.resize(cropping, dsize = (w, h), interpolation = cv2.INTER_LINEAR)
                # plt.imshow(crop)
                # plt.show()

                x(crop/255)
                y(label)
        
        ''' 견종별로 따로 따로 저장 '''
        images = np.array(X)
        target = np.array(Y)

        np.save('./data/face_image_%s.npy'%(folder), images)
        np.save('./data/face_label_%s.npy'%(folder), target)
        print('---------- END %s ---------'%(folder))

    
face_detector(train_path, 300, 300)


''' 따로 따로 저장한 견종 numpy로 합치기'''
def _append(image_label, shape):                     # shape : 저장한 파일 shape / 행은 무조건 0
    arr = np.empty(shape, np.float32)
    folder_name = os.listdir(train_path)

    for f in folder_name:
        y = np.load('./data/face_%s_%s.npy'%(image_label, f))
        print(y.shape)
        arr = np.append(arr, y, axis = 0)

    print(arr.shape)

    np.save('./data/face_%s_part1.npy'%(image_label), arr)
    print('------------- save complete ---------------')

# image파일 합칠시 'image'
# label파일 합칠시 'label'
_append('image',(0, 300, 300, 3))
_append('label', (0, ))

end = time.time()
print('Time :', end - start)



