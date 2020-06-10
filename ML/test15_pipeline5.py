from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size = 0.8, random_state = 42
)

model = DecisionTreeClassifier(max_depth =4)      # max_depth 몇 이상 올라가면 구분 잘 못함
                                                  
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print('acc: ', acc)

print(model.feature_importances_)
'''
acc:  0.9473684210526315
[0.         0.03139487 0.         0.         0.         0.
 0.         0.70458252 0.         0.         0.         0.
 0.         0.01221069 0.         0.         0.02530295 0.0162341
 0.         0.         0.05329492 0.02819607 0.05247428 0.
 0.00940897 0.         0.         0.06690062 0.         0.        ]
'''