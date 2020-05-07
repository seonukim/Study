# 1. 사이킷런 패키지의 로지스틱 회귀 알고리즘으로 iris 데이터 분류하기
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = 'liblinear', multi_class = 'auto',
                        C = 100.0, random_state = 1)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier = lr,
                      test_idx = range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

# 2. 훈련 샘플이 어떤 클래스에 속할 확률 계산하기
lr.predict_proba(X_test_std[:3, :])
# lr.predict_proba(X_test_std[:3, :]).sum(axis = 1) 열을 모두 더했을 때 1이 되는지 확인함

# 3. 클래스 레이블 확인(행에서 가장 큰 값의 열이 예측 클래스 레이블)
lr.predict_proba(X_test_std[:3, :]).argmax(axis = 1)
lr.predict(X_test_std[:3, :])

# 4. 하나의 행을 2차원 포맷으로 변경하기
lr.predict(X_test_std[0, :].reshape(1, -1))

# 5. overfitting 방지, 규제: L2 규제 - 시각화
weight, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(solver = 'liblinear',
                            multi_class = 'auto',
                            C = 10. ** c, random_state = 1)
    lr.fit(X_train_std, y_train)
    weight.append(lr.coef_[1])
    params.append(10. ** c)
weight = np.array(weight)
plt.plot(params, weight[:, 0],
         label = 'petal length')
plt.plot(params, weight[:, 1],
         linestyle = '--',
         label = 'petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc = 'upper left')
plt.xscale('log')
plt.show()