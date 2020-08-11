# 뉴럴 네트워크
# nn.Module은 여러 개의 레이어와 output을 리턴하는 forward(input) 메소드를 포함한다
# 일반적인 뉴럴 네트워크의 학습절차는 다음과 같다
# 학습 가능한 파라미터나 weight가 있는 뉴럴 네트워크를 정의한다
# 입력 데이터셋에 대한 반복 학습을 진행한다
# 네트워크를 통해 입력 값을 처리한다
# loss를 계산한다 (loss = output - target)
# 그라디언트를 네트워크의 파라미터로 역전파시킨다
# 다음의 공식으로 weight를 갱신한다
# weight = weight - learning_rate * gradient

# 네트워크 정의
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5 * 5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation : y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # MaxPooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
'''
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)       
  (fc2): Linear(in_features=120, out_features=84, bias=True)        
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
'''
# 사용자는 직접 forward 함수만 정의해주면 된다
# 그라디언트가 계산되는 backward 함수는 autograd를 사용함으로써
# 자동으로 정의된다
# 모델의 학습 가능한 파라미터는 net.parameters()에 의해 리턴된다
net.parameters()

# 32 x 32 크기의 랜덤 값을 입력으로 사용하면 다음과 같다
# 참고로 이 네트워크(LeNet, 현대의 CNN 네트워크의 시초라고 할 수 있음)에 대한 예상 입력 크기는 32x32이다
# 이 네트워크를 MNIST 데이터셋을 대상으로 사용하기 위해서는
# 데이터셋의 이미지를 32x32 크기로 변경할 필요가 있다
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
'''
tensor([[-0.0598, -0.1047,  0.0454, -0.0633, -0.0051, -0.0847,  0.1367,  0.0842,
          0.0749,  0.1146]], grad_fn=<ThAddmmBackward>)
'''

# 모든 파라미터의 그라디언트 버퍼를 0으로 설정하고, 랜덤 그라디언트로 역전파를 수행한다
net.zero_grad()
out.backward(torch.randn(1, 10))

# torch.nn은 mini-batch만을 지원한다
# 전체 torch.nn 패키지는 mini-batch 형태인 입력만을 지원하며
# 단일 데이터는 입력으로 지원하지 않는다
# 예를 들어 nn.Conv2d는 nSamples x nChannels x Height x Width의 4차원 텐서를 취한다
# 만약 단일 샘플(1개의 데이터)이 있다면, input.unsqueeze(0)를 사용하여 가짜 임시 배치 차원을 추가하면 된다

# torch.Tensor - backward()와 같은 autograd 연산을 지원하는 다차원 배열이며 텐서에 대한 그라디언트를 가지고 있다
# nn.Module - 뉴럴 네트워크 모듈로서 파라미터를 GPU로 옮기거나, 내보내기, 불러오기 등의 보조 작업을 이용하여
# 파라미터를 캡슐화하는 편리한 방법이다
# nn.Parameter - 텐서의 일종. Module에 속성을 할당할 때 파라미터로 자동 등록된다
# autograd.Function - autograd 연산의 forward와 backward에 대한 정의를 구현한다
# 모든 Tensor 연산은 최소한 하나의 Function 노드를 생성하는데, 이 노드는 Tensor를 생성하고
# 기록을 인코딩하는 여러 함수들에 연결된다


## 손실함수(Loss Function)
# 오차 함수는 (출력, 정답) 형태의 입력을 받아
# 출력이 정답에서 얼마나 멀리 떨어져 있는지 추정하는 값을 계산한다
# nn 패키지에는 여러가지 손실함수들이 존재하는데 가장 간단한 loss는
# nn.MSELoss로 입력과 정답 사이의 평균 제곱 오차(Mean-Squared-Error)를 계산한다
output = net(input)
target = torch.arange(1, 11)        # a dummy target, for example
target = target.view(1, -1)         # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

# 이제, 만약 사용자가 .grad_fn 속성을 이용하여 역방향으로 loss를 따라가려면 다음과 같은 계산 그래프를 볼 수 있다
# 따라서 loss.backward()를 호출하면 전체 그래프는 손실에 대하여 미분 계산이 수행되며,
# requires_grad = True인 그래프 내의 모든 텐서들은 그라디언트로 누적된 .grad 텐서를 갖게 된다
print(loss.grad_fn)     # MSELoss
print(loss.grad_fn.next_functions[0][0])    # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])   # ReLU

## 역전파(Backprop)
net.zero_grad()         # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

## 가중치 갱신(optimizer)
# 기본적으로 가장 간단한 갱신 룰은 Stochastic Gradient Descent(SGD)이다
# weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# PyTorch에서는 다양한 갱신 룰을 포함하는 torch.optim 패키지를 제공한다
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# in your training loop:
optimizer.zero_grad()       # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()        # Does the update