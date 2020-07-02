import numpy as np

x = np.array([0,1,2,3,4,5])
y = np.array([6,9,12,15,18,21])

N = len(x)
반복횟수 = 3000
learning_rate = 0.01
기울기 = 0
절편 = 0

for i in range(반복횟수) :
    
    # 1. 함수정의
    hypothesis = 기울기*x + 절편
    
    # 2. 비용함수(RSS) 정의
    cost = (1/N)*np.sum((y-hypothesis)**2)
    
    # 3. 비용함수를 최소화하는 경사하강법을 통해 기울기와 절편 편미분 계산
    gradient_기울기 = -(2/N)*np.sum(x*( y - hypothesis ))
    gradient_절편 = -(2/N)*np.sum(( y - hypothesis ))
    
    # 4. learning_rate는 학습률(학습속도)를 적용하여 기울기, 절편 값 업데이터 및 보정
    기울기 = 기울기 - (gradient_기울기 * learning_rate)
    절편 = 절편 - (gradient_절편 * learning_rate)
    
    if i % 100 == 0 :
        print('반복횟수: {}, cost: {:.3f}, 기울기: {:.3f}, 절편: {:3f}'.format(i, cost, 기울기, 절편), end='\n\n')
        
print('기울기: {:.3f}'.format(기울기))
print('절편: {:.3f}'.format(절편), end='\n\n')

print('경사하강법으로 구한 w1, w0으로 실제값(y)과 얼마나 차이가 나는지 출력: ')
print(기울기*x + 기울기)