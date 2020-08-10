from __future__ import print_function
import torch
import warnings ; warnings.filterwarnings(action='ignore')

# 텐서(Tensors)
# Tensors는 Numpy의 ndarrays와 유사한 것으로 계산 속도를 빠르게 하기 위해
# GPU에서 사용할 수 있는 것이라 보면 된다.

# 초기화 되지 않은 5*3 행렬 생성하기
x = torch.empty(5, 3)
print(x)

# 랜덤으로 초기화된 행렬 생성하기
x = torch.randn(5, 3)
print(x)

# 0으로 채워지고 long 데이터 타입을 가지는 행렬 생성하기
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 데이터로부터 직접 텐서 생성하기
x = torch.tensor([5.5, 3])
print(x)

# 혹은, 이미 존재하는 텐서를 기반으로 새로운 텐서를 생성할 수도 있다
# 아래 코드와 같은 방법들은 입력 텐서의 속성(데이터 타입 등)들이 사용자에 의해 새롭게 제공되지 않는 이상
# 기존의 값들을 사용한다
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

# 생성한 텐서의 사이즈 확인하기
print(x.size())            # torch.Size([5, 3])
# torch.Size()는 파이썬의 자료형이기 때문에 튜플과 관련된 모든 연산을 지원한다

## 연산(Operation)
# 더하기 (1)
y = torch.rand(5, 3)
print(x + y)

# 더하기 (2)
print(torch.add(x, y))

# 더하기 ; 파라미터로 결과가 저장되는 결과 텐서 이용
res = torch.empty(5, 3)
torch.add(x, y, out=res)
print(res)

# 더하기 ; 제자리(in-place)
# adds x to y
y.add_(x)
print(y)

# 텐서를 제자리에서 변조하는 연산은 _ 문자를 이용하여 postfix(연산자를 피연산자의 뒷쪽에 표시)로 표기한다.
# 예를 들면, x.copy_(y)와 x.t_()는 x를 변경시킨다.
# Numpy와 같이 표준 인덱싱을 사용할 수도 있다.
print(x[:, 1])          # tensor([ 1.3982,  1.2788,  0.5974, -0.5015,  1.0546])

# 사이즈 변경(Resizing) : 텐서의 사이즈를 변경하거나, 모양(shape)을 변경하고 싶다면
# torch.view()를 사용한다
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())         # torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])

# 1개의 요소를 가지는 텐서(one element tensor)가 있다면, .item()을 사용하여
# Python에서의 숫자 데이터 값처럼 값을 얻을 수 있다
x = torch.randn(1)
print(x)
print(x.item())

'''
전치, 인덱스, 슬라이스, 수학계산, 선형대수, 랜덤넘버 등등 기타 텐서 연산은 다음 링크 참고
http://pytorch.org/docs/stable/torch.html
'''

## Numpy 변환(Numpy Bridge)
# 토치 텐서를 Numpy 배열로 바꾸거나, 반대로 하고 싶은 경우
# 토치 텐서와 Numpy 배열은 기본적으로 메모리 위치를 공유하기 때문에 하나를 변경하면 다른 하나도 변경된다
# 토치 텐서를 Numpy 배열로 변경
a = torch.ones(5)
print(a)            # tensor([1., 1., 1., 1., 1.])

# 배열로 변경
b = a.numpy()
print(b)            # [1. 1. 1. 1. 1.]

# Numpy 배열의 값이 어떻게 변하는지?
a.add_(1)
print(a)            # tensor([2., 2., 2., 2., 2.])
print(b)            # [2. 2. 2. 2. 2.]

# Numpy 배열을 토치 텐서로 변경
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)            # [2. 2. 2. 2. 2.]
print(b)            # tensor([2., 2., 2., 2., 2.], dtype=torch.float64)

# CUDA 텐서
# 텐서는 .to()를 이용하여 CUDA를 지원하는 그 어떠한 디바이스로도 옮길 수 있다
if torch.cuda.is_available():
    device = torch.device('cuda')           # a CUDA device object
    y = torch.ones_like(x, device=device)   # directly create a tensor on GPU
    x = x.to(device)                        # or just use strings ``.to('cuda')``
    z = x + y
    print(z)
    print(z.to('cpu', torch.double))        # ``.to()`` can also change dtype together!
'''
Result
tensor([0.1809], device='cuda:0')
tensor([0.1809], dtype=torch.float64)
'''