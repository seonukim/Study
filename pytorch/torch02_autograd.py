# Autograd : 자동 미분(Automatic Differentiation)
# Pytorch에서 뉴럴 네트워크의 중심은 autograd라는 패키지라 할 수 있다
# autograd 패키지는 텐서의 모든 연산에 대하여 자동 미분을 제공한다
# 이 패키지는 실행 시점에 정의되는 (define-by-run) 프레임워크인데
# 다시 말하면 코드가 어떻게 실행되는지에 따라 역전파(backpropagation)이 정의되며,
# 각각의 반복마다 역전파가 달라질 수 있다는 것이다.

# 텐서(Tensors)
# torch.Tensor는 패키지에서 가장 중심이 되는 클래스
# 텐서의 속성 중 하나인 .requires_grad를 True로 세팅하면
# 텐서의 모든 연산에 대하여 추적을 시작한다
# 계산 작업이 모두 수행되었다면, .backward()를 호출하여 모든 그라디언트들을 자동으로 계산할 수 있다
# 이 텐서를 위한 그라디언트는 .grad 속성에 누적되어 저장된다

# 텐서에 대하여 기록 추적을 중지하려면 .detach()를 호출하여 현재의 계산 기록으로부터 분리시키고
# 이후에 일어나는 계산들은 추적되지 않게 할 수 있다

# 기록 추적 및 메모리 사용에 대하여 방지를 하려면
# 코드 블럭을 with torch.no_grad(): 로 래핑할 수 있다
# 이는 특히 모델을 평가할 때 엄청난 도움이 되는데, 왜냐하면 모델은 requires_grad = True 속성이 적용된
# 학습 가능한 파라미터를 가지고 있을 수 있으나 우리는 그라디언트가 필요하지 않기 때문이다

# 자동 미분을 위해 매우 중요한 클래스가 하나 더 있는데 바로 Function이다

# Tensor와 Function은 상호 연결되어 있으며, 비순환(비주기) 그래프를 생성하는데,
# 이 그래프는 계산 기록 전체에 대하여 인코딩을 수행한다
# 각 변수는 Tensor를 생성한 Function을 참조하는 .grad_fn 속성을 가지고 있지만
# 사용자에 의해 생성된 텐서는 제외한다 - 해당 텐서들은 grad_fn 자체가 None 상태이다

# 만약 도함수(derivatives)를 계산하고 싶다면, Tensor의 .backward()를 호출한다
# 만약 Tensor가 스칼라(i.e.한개의 요소를 가지고 있는) 형태라면, backward()사용에 있어
# 그 어떠한 파라미터도 필요하지 않다
# 그러나 한 개 이상의 요소를 가지고 있다면 올바른 모양(matching shape)의 텐서인 gradient 파라미터를 명시할 필요가 있다

import torch

# 텐서를 생성하고 requires_grad = True로 세팅하여 계산을 추적한다
x = torch.ones(2, 2, requires_grad = True)
print(x)

# 텐서에 대하여 임의의 연산을 수행한다
y = x + 2
print(y)

# y는 연산의 결과로써 생성된 것이기 때문에 grad_fn을 가지고 있다
print(y.grad_fn)            # <AddBackward object at 0x0000023D2FF35C88>

# y에 대하여 임의의 연산을 수행한다
z = y * y * 3
out = z.mean()
print(z,'\n',out)
'''
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward>)
tensor(27., grad_fn=<MeanBackward1>)
'''

# .requires_grad_(...)은 이미 존재하는 텐서의 requires_grad 플래그를 제자리(in-place)에서 변경한다
# 입력 플래그를 명시하지 않은 경우 기본적으로 True가 된다
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)      # False
a.requires_grad_(True)
print(a.requires_grad)      # True
b = (a * a).sum()
print(b.grad_fn)            # <SumBackward0 object at 0x000002B4A1A264E0>

## 그라디언트(Gradients)
# out은 하나의 스칼라(single scalar) 값을 가지고 있기 때문에
# out.backward()는 out.backward(torch.tensor(1))와 동등한 결과를 리턴한다
out.backward()
print(x.grad)
'''
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
'''

# autograd를 이용하여 여러가지 연산 수행해보기
x = torch.randn(3, requires_grad = True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)            # tensor([605.8693, 815.9294,  66.2642], grad_fn=<MulBackward>)

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)           # tensor([ 204.8000, 2048.0000,    0.2048])

# 또한 코드 블럭을 with torch.no_grad(): 로 래핑하여 requires_grad = True 속성이 명시된
# autograd가 텐서의 추적 기록에 남지 않게 할 수 있다
print(x.requires_grad)              # True
print((x**2).requires_grad)         # True

with torch.no_grad():
    print((x**2).requires_grad)     # False

'''
추후 시간이 있다면 autograd와 Function에 대한 문서를 읽어보자
http://pytorch.org/docs/stable/autograd.html
'''