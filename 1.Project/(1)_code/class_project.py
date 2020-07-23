import numpy as np
import cv2, os, dlib

class Project:                                              # 해당 경로의 파일들 처리
    def __init__(self):
        pass

    def face_detector(self, img, detector_path):

        self.img_face = img.copy()

        detector = dlib.cnn_face_detection_model_v1(detector_path)
        self.dets = detector(self.img_face, upsample_num_times = 1)
        
        return self.dets

    def BBox(self, number, dets):
        d = dets
        # print("Detection {}: Left:{} Top:{} Right:{} Bottom:{} Confidence:{}".format(number+1, 
        #         d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

        x1, y1 = d.rect.left(), d.rect.top()
        x2, y2 = d.rect.right(), d.rect.bottom()

        #-------- bbox 키우기-----------
        pad = (x2 - x1)
        x1 = x1 - (pad / 4)
        y1 = y1 - (pad * 3/8) 
        x2 = x2 + (pad / 4)
        y2 = y2 + (pad / 8)

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0

        x1, x2, y1, y2 = map(int, (x1, x2, y1, y2))     # int형으로 변환
        self.bbox = [x1, x2, y1, y2]
            
        img_bbox = self.img_face.copy()
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), thickness = 1, color = (255, 122, 122), lineType = cv2.LINE_AA)
                
        self.img_bbox = img_bbox                        # image
        # cv2.imshow('BBox',self.img_bbox)
        # cv2.waitKey(0)
        # cv2.destroyWindow('BBox')
        
        return self.bbox        # 수정한 bounding box 좌표

    def crop(self):
        x1 = self.bbox[0]       # x1
        x2 = self.bbox[1]       # x2
        y1 = self.bbox[2]       # y1    
        y2 = self.bbox[3]       # y2

        self.img_crop = self.img_face[y1:y2, x1:x2]
        # cv2.imshow('Crop', self.img_crop)
        # cv2.waitKey(0)
        # cv2.destroyWindow('Crop')

        return self.img_crop    # 자른 이미지








    





        
        


