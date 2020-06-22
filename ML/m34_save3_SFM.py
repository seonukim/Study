# 200622_25
# iris / model save


import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score


### 데이터 ###
x, y = load_iris(return_X_y=True)
print(x.shape)      # (150, 4)
print(y.shape)      # (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)


### 기본 모델 ###
model = XGBClassifier(objective='multi:softmax', n_estimators=300, learning_rate=0.1, n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('acc :', score)

#== Default acc : 0.9 ==#


### feature engineering ###
thresholds = np.sort(model.feature_importances_)
print(thresholds)
'''
[0.01922086 0.02886017 0.3315531  0.6203659 ]
'''

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    selection_model = XGBClassifier(objective='multi:softmax', n_estimators=300, learning_rate=0.1, n_jobs=-1)
    selection_model.fit(select_x_train, y_train, verbose=True, eval_metric=['mlogloss', 'merror'],
                eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                early_stopping_rounds=20)
    
    y_pred = selection_model.predict(select_x_test)

    acc = accuracy_score(y_test, y_pred)

    print('Thresh=%.3f, n=%d, acc: %.2f%%' %(thresh, select_x_train.shape[1], acc*100.0))
    model.save_model('./model/xgb_save/iris_n=%d_acc=%.3f.model' %(select_x_train.shape[1], acc))

# Thresh=0.019, n=4, acc: 100.00%
# Thresh=0.029, n=3, acc: 100.00%
# Thresh=0.332, n=2, acc: 100.00%
# Thresh=0.620, n=1, acc: 93.33%