import numpy as np
import cv2, os, dlib

class Project:                                  # 해당 경로의 파일들 처리
    def __init__(self, path):
        self.top_path = path
        self.category = os.listdir(top_path)    # list로 반환
        pass

    # 이미지 불러오기
    def load_image(self):
        img_b = []
        for folder in (self.category):                      # folder 이름
            self.image_dir = self.top_path + '/'+folder+'/' # 찾을 image 경로

            for top, dir, f in os.walk(image_dir):          # image 이름 탐색
                for filename in f:
                    img = cv2.imread(self.image_dir+filename)
                    self.img_b.append(img)

        return self.img_b

    def face_detector(self, detector_path):
        img = cv2.cvtColor(self.img_b, cv2.COLOR_BGR2RGB)

        self.img_copy = img.copy()

        detector = dlib.cnn_face_detection_model_v1(detector_path)
        self.dets = detector(img, upsample_num_times=1)
        
        return self.dets

    def bbox(self):
        bbox = []

        d = self.dets
        print("Detection {}: Left:{} Top:{} Right:{} Bottom:{} Confidence:{}".format(i+1, 
                d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

        x1, y1 = d.rect.left(), d.rect.top()
        X2, Y2 = d.rect.right(), d.rect.bottom()

        #-------- bbox 키우기-----------
        pad = (x2 - x1)
        x1 = x1 - (pad/4)
        y1 = y1 - (pad*3/8) 
        x2 = x2 + (pad/4)
        y2 = y2 + (pad/8)

        if x1 < 0:
            x1 = 0
        elif y1 < 0:
            y1 = 0

        x1, x2, y1, y2 = map(int, (x1, x2, y1, y2))     # int형으로 변환
        self.bbox = bbox.append([x1, x2, y1, y2])

        img_bbox = self.img_copy
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), thickness=1, color=(255, 122, 122), lineType=cv2.LINE_AA)
            
        self.img_bbox = img_bbox                            # image
        cv2.imshow('BBox',self.img_bbox)
        cv2.waitKey(0)
        
        return  self.bbox

    def crop(self):
        x1 = self.bbox[0]
        x2 = self.bbos[1]
        y1 = self.bbox[2]
        y2 = self.bbox[3]

        self.img_crop = self.img_copy[y1:y2, x1:x2]
        cv2.imshow('Crop', self.img_crop)
        cv2.waitKey(0)

        return self.img_crop

    def labeling(self):
        num_class = len(self.category)
        labeling = []

        for idex, folder in enumerate(self.category):
            label = [0 for i in range(num_class)]
            label[idex] = 1

            labeling.append(label)
        
        return labeling






    





        
        


