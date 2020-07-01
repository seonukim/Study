import os
import numpy as np
import struct
import matplotlib.pyplot as plt
path = 'C:/Users/bitcamp/Desktop/서누/Review/mnist/'
os.chdir(path)

## mnist 가져오는 함수
def load_mnist(path, kind = 'train'):
    labels_path = os.path.join('C:/Users/bitcamp/Desktop/서누/Review/mnist/',
                               '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join('C:/Users/bitcamp/Desktop/서누/Review/mnist/',
                               '%s-images.idx3-ubyte' % kind)
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype = np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype = np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2         # mnist의 픽셀 값을 -1에서 1 사이로 정규화
    return images, labels

'''
> : 빅 엔디언(big-endian)을 나타낸다, 바이트가 저장된 순서를 정의
I : 부호 없는 정수를 의미함
'''

## mnist 데이터 불러오기
x_train, y_train = load_mnist('', kind = 'train')
print(f'행: {x_train.shape[0]}, 열: {x_train.shape[1]}')        # 행: 60000, 열: 784

x_test, y_test = load_mnist('', kind = 't10k')
print(f'행: {x_test.shape[0]}, 열: {x_test.shape[1]}')          # 행: 10000, 열: 784


## mnist 이미지 샘플
fig, ax = plt.subplots(nrows = 2, ncols = 5, sharex = 'all', sharey = 'all')
ax = ax.flatten()
for i in range(10):
    img = x_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap = 'Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.show()

## mnist 이미지 중 숫자 7의 처음 25개 샘플
fig, ax = plt.subplots(nrows = 5, ncols = 5, sharex = 'all', sharey = 'all')
ax = ax.flatten()
for i in range(25):
    img = x_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap = 'Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.show()


## numpy의 savez_compressed를 사용하여 .npz로 저장하기
# np.savez_compressed(path + 'mnist_scaled.npz',
#                     x_train = x_train, y_train = y_train,
#                     x_test = x_test, y_test = y_test)
'''
numpy의 saves함수는 파이썬 내장함수인 pickle 모듈과 비슷하지만, 넘파이 배열을 저장하는데 최적화
saves함수는 데이터를 압축하여 .npy 포맷 파일을 담고 있는 .npz 파일을 생성한다
위 코드에서는 saves 대신에 saves_compressed를 사용했는데, 이 함수는 savez와 사용법은 같지만
파일 크기를 더 작게 압축한다(약 400MB를 22MB 정도로 줄임)
'''

## mnist 파일 불러오기
mnist = np.load(path + 'mnist_scaled.npz')
print(mnist.files)          # ['x_train', 'y_train', 'x_test', 'y_test']

x_train, y_train, x_test, y_test = [mnist[f] for f in mnist.files]
print(f'"X"; train: {x_train.shape[0]}, {x_train.shape[1]}')
print(f'"X"; test: {x_test.shape[0]}, {x_test.shape[1]}')