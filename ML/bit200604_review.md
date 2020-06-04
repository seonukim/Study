## BIT 2020.06.04 Review
___

#### 머신러닝
	머신러닝 모델의 종류
    - Support Vector Machine (SVM)
    - K-Nearest Neighbor (KNN); K 최근접 이웃
    - RandomForest; 앙상블
___


 **SVM**
서포트 벡터 머신은 퍼셉트론의 확장으로 생각할 수 있다. SVM의 최적화 대상은 `마진`을 최대화 하는 것인데, 여기서 `마진`이란 클래스를 구분하는 결정 경계와 이 경계에서 가장 가까운 **Train sample** 사이의 거리라고 정의한다. 이런 샘플을 **서포트 벡터**라고 한다.
최대 마진(large margin; 큰 마진)의 결정 경계를 원하는 이유는 **일반화 오차가 낮아지는 경향**이 있기 때문이다. 반면, 작은 마진의 모델은 `overfitting`의 위험에 노출되기 쉽다.


**SVM 예시코드**

```python
from keras.utils import np_utils
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# 1. iris 데이터 로드
x, y = load_iris(retusn_X_y = True)
print(x.shape)				# (150, 4)
print(y.shape)				# (150,)

# 2. 데이터 split
x_train, x_test, y_train, y_test = train_test_split(
	x, y, test_size = 0.2)
print(x_train.shape)		# (120, 4)
print(x_test.shape)		 # (30, 4)
print(y_train.shape)		# (120,)
print(y_test.shape)		 # (30,)

# 3. OneHotEncoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)		# [0. 1. 0.]
print(y_test.shape)		 # [0. 1. 0.]

# 4. 모델링 및 훈련
model = SVC(kernel = 'linear', C = 1.0, randam_state = 1)
model.fit(x_train, y_train)

# 5. 모델 평가 및 결과 예측
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("모델 정확도 : ", acc)		# 0.991666666666667
```
___

**KNN 최근접이웃**
k-최근접 이웃 알고리즘은 전형적인 **게으른 학습기(lazy learner)** 이다. 단순하기에 게으르다고 하는 것이 아니라, 이 알고리즘은 훈련데이터에서 판별 함수를 학습하는 대신 훈련 데이터셋을 **메모리에 저장**하기 때문이다.
KNN 알고리즘은 다음과 같은 단계로 요약할 수 있다.
1. 숫자 k와 거리 측정 기준을 선택
2. 분류하려는 샘플에서 k개의 최근접 이웃을 찾는다.
3. **다수결 투표**를 통해 클래스 레이블을 반환한다.

![KNN](https://github.com/seonukim/Study/blob/master/ML/p130.jpg)

[그림. k-최근접 이웃의 다수결 투표]

**KNN 예시 코드**
```python
from keras.utils import np_utils
from sklearn.svm import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# 1. iris 데이터 로드
x, y = load_iris(retusn_X_y = True)
print(x.shape)				# (150, 4)
print(y.shape)				# (150,)

# 2. 데이터 split
x_train, x_test, y_train, y_test = train_test_split(
	x, y, test_size = 0.2)
print(x_train.shape)		# (120, 4)
print(x_test.shape)		 # (30, 4)
print(y_train.shape)		# (120,)
print(y_test.shape)		 # (30,)

# 3. OneHotEncoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)		# [0. 1. 0.]
print(y_test.shape)		 # [0. 1. 0.]

# 4. 모델링 및 훈련
model = KNeighborsClassifier(n_neighbors = 5, p = 2, metrics = 'minkowski)
model.fit(x_train, y_train)

# 5. 모델 평가 및 결과 예측
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("모델 정확도 : ", acc)		# 0.956140350877193
```
___

**RandomForest**
**랜덤 포레스트(Random forest)**는 뛰어난 분류 성능, 확장성, 쉬운 사용법을 가진 강력한 모델이다. 랜덤 포레스트는 **Decision Tree(의사결정나무)** 여러 개를 묶은 `앙상블(ensemble)`로 이해할 수 있다. 랜덤 포레스트 이면의 아이디어는 여러 개의 깊은 의사결정나무를 평균 내는 것이다. 랜덤 포레스트는 다음과 같은 네 단계로 요약할 수 있다.
1. n개의 랜덤한 **부트스트랩(bootstrap)** 샘플을 뽑는다.(중복 허용)
2. 부트스트랩 샘플에서 결정 트리를 학습한다. 각 노드에서 다음과 같다.
	- 중복을 허용하지 않고 랜덤하게 d개의 특성을 선택
	- 정보 이득과 같은 목적 함수를 기준으로 최선의 분할을 만드는 특성을 사용해서 노드를 분할한다.
3. 단계 1 ~ 2를 k번 반복한다.
4. 각 트리의 예측을 모아 **다수결 투표**로 클래스 레이블을 할당한다.

**RandomForest 예시 코드**
```python
from keras.utils import np_utils
from sklearn.svm import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# 1. iris 데이터 로드
x, y = load_iris(retusn_X_y = True)
print(x.shape)				# (150, 4)
print(y.shape)				# (150,)

# 2. 데이터 split
x_train, x_test, y_train, y_test = train_test_split(
	x, y, test_size = 0.2)
print(x_train.shape)		# (120, 4)
print(x_test.shape)		 # (30, 4)
print(y_train.shape)		# (120,)
print(y_test.shape)		 # (30,)

# 3. OneHotEncoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)		# [0. 1. 0.]
print(y_test.shape)		 # [0. 1. 0.]

# 4. 모델링 및 훈련
model = RandomForestClassifier(criterion = 'gini',
							   n_estimator = 25,
                               random_state = 1,
                               n_jobs = 2)
model.fit(x_train, y_train)

# 5. 모델 평가 및 결과 예측
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("모델 정확도 : ", acc)		# 0.956140350877193
```
















