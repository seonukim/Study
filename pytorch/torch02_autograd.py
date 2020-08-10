# Autograd : 자동 미분(Automatic Differentiation)
# Pytorch에서 뉴럴 네트워크의 중심은 autograd라는 패키지라 할 수 있다
# autograd 패키지는 텐서의 모든 연산에 대하여 자동 미분을 제공한다
# 이 패키지는 실행 시점에 정의되는 (define-by-run) 프레임워크인데
# 다시 말하면 코드가 어떻게 실행되는지에 따라 역전파(backpropagation)이 정의되며,
# 각각의 반복마다 역전파가 달라질 수 있다는 것이다.