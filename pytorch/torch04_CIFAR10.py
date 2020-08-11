## CIFAR-10 데이터셋을 활용한 PyTorch 이미지 분류기 학습
## 순서
# 1. torchvision을 이용하여 CIFAR-10의 학습, 평가 데이터셋을 로드하여 정규화한다
# 2. 컨볼루션 뉴럴 네트워크를 정의한다
# 3. 손실함수를 정의한다
# 4. 학습 데이터를 이용하여 네트워크를 학습시킨다
# 5. 평가 데이터셋을 이용하여 네트워크를 평가한다

import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 데이터셋 로드 및 정규화
import torch
import torchvision
import torchvision.transforms as transforms

# torchvision 패키지의 출력은 [0, 1] 범위의 PILImage 이미지이다
# 출력 데이터를 범위 [-1, 1]의 텐서로 정규화 시킨다
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
trainset = torchvision.datasets.CIFAR10(root='../cifar10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
'''
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnomalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''

# CNN 모델링
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Loss function 및 Optimizer 정의
# Loss function은 Classfication Cross-Entropy를 사용하고
# Optimizer는 momentum을 세팅한 SGD를 사용한다
import torch.optim as op
criterion = nn.CrossEntropyLoss()
optimizer = op.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 학습
for epoch in range(2):      # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:        # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % \
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')