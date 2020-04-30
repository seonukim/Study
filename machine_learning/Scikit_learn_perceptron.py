# 1. 패키지 호출하기
from sklearn import datasets
import numpy as np

# 2. 붓꽃 데이터 및 정수 레이블 호출
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('클래스 레이블: ', np.unique(y))
# result : 클래스 레이블: [0 1 2]

# 3. 훈련 셋과 테스트 셋으로 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

# 4. stratify를 통한 계층화 확인하기
print('y의 레이블 카운트: ', np.bincount(y))
print('y_train의 레이블 카운트: ', np.bincount(y_train))
print('y_test의 레이블 카운트: ', np.bincount(y_test))
# result : y의 레이블 카운트: [50 50 50]
#          y_train의 레이블 카운트: [35 35 35]
#          y_test의 레이블 카운트: [15 15 15]

# 5. 특성 스케일 조정: preprocessing모듈의 StandardScaler - 특성 표준화
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 6. Perceptron 모델 훈련
# eta0 : 학습률, max_iter : 에포크 횟수(최대 반복 횟수)
from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter = 40, eta0 = 0.1, tol = 1e-3, random_state = 1)
ppn.fit(X_train_std, y_train)

# 7. 예측
# 오차 계산
y_pred = ppn.predict(X_test_std)
print('잘못 분류된 샘플 개수: %d' % (y_test != y_pred).sum())
# result : 잘못 분류된 샘플 개수: 1

# 분류 정확도 계산(1 - 오차)
from sklearn.metrics import accuracy_score
print('정확도: %.2f' % accuracy_score(y_test, y_pred))
print('정확도: %.2f' % ppn.score(X_test_std, y_test))
# result : 정확도 : 0.98
#          정확도 : 0.98

# 8. 붓꽃 샘플 시각화
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# plot_decision_regions 함수 정의하기
def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):

    # 마커와 컬러맵 설정
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1],
                    alpha = 0.8, c = colors[idx],
                    marker = markers[idx], label = cl,
                    edgecolors = 'black')

    # 테스트 샘플을 부각하여 그리기
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c = '', edgecolors = 'black', alpha = 1.0,
                    linewidth = 1, marker = 'o',
                    s = 100, label = 'test set')

# 9. 결과 그래프에 표시할 테스트 샘플 인덱스 저장하기
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X = X_combined_std,
                      y = y_combined,
                      classifier = ppn,
                      test_idx = range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

# result image
![iris_perceptron.png](/Users/seonwoo/Desktop/vscode image/iris_perceptron.png)
