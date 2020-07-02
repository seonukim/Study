import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 - 4*x + 6
x = np.linspace(-1, 6, 100)
y = f(x)

# 그리자!
plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
# plt.show()


gradient = lambda x: 2*x - 4
'''
위 f로 정의한 2차 함수를 미분하여 gradient 1차 함수로 만듦
왜? 2차함수의 기울기가 0이 되는 지점(전역최소비용)을 찾기 위해서
'''


x0 = 0.0
maxIter = 10
learning_rate = 0.25

print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

for i in range(maxIter):
    x1 = x0 - learning_rate * gradient(x0)
    x0 = x1

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))
'''
위 반복문은 경사하강법에서 설정한 학습률에 따라
함수의 최소 비용 지점을 찾아가는 과정을 보여준다
'''
