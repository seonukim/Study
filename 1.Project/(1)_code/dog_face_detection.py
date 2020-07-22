import dlib, cv2, os
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt

# load models
detector = dlib.cnn_face_detection_model_v1('./project/project02/weight/dogHeadDetector.dat')
predictor = dlib.shape_predictor('./project/project02/weight/landmarkDetector.dat')

# img_path = './project/project02/data/multi_dog_face.jpg'
img_path = './project/project02/data/dog_face.jpg'

filename, ext = os.path.splitext(os.path.basename(img_path))
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(16, 16))
# plt.imshow(img)
# plt.show()

dets = detector(img, upsample_num_times=1)

img_result = img.copy()

for i, d in enumerate(dets):
    print("Detection {}: Left:{} Top:{} Right:{} Bottom:{} Confidence:{}".format(i, 
            d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

    s = 30
    x1, y1 = d.rect.left(), d.rect.top()
    x2, y2 = d.rect.right(), d.rect.bottom()

    cv2.rectangle(img_result, (x1, y1-s), (x2+s, y2+s), thickness=2, color=(122, 122, 122), lineType=cv2.LINE_AA)
    print('1')

# plt.figure(figsize=(16, 16))
# plt.imshow(img_result)
# plt.show()

print('-----------')

shapes = []

for i, d in  enumerate(dets):
    shape = predictor(img, d.rect)
    shape = face_utils.shape_to_np(shape)

    for i, p in enumerate(shape):
        shapes.append(shape)
        cv2.circle(img_result, center=tuple(p), radius=3, color=(0,0, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(img_result, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    print('2')

img_out = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
cv2.imwrite('img/%s_out%s'%(filename, ext), img_out)
plt.figure(figsize=(16, 16))
plt.imshow(img_result)
plt.show()


