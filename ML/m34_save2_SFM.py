# 200622_25
# cancer / model save


import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score


### 데이터 ###
x, y = load_breast_cancer(return_X_y=True)
print(x.shape)      # (569, 30)
print(y.shape)      # (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)


### 기본 모델 ###
model = XGBClassifier(n_estimators=300, learning_rate=0.1, n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('acc :', score)

#== Default acc : 0.9736842105263158 ==#


### feature engineering ###
thresholds = np.sort(model.feature_importances_)
print(thresholds)
'''
[0.00046491 0.0013246  0.00139511 0.00200115 0.00204048 0.00244829
 0.00346774 0.00359193 0.00407993 0.00500716 0.00536249 0.00552303
 0.00699229 0.00727135 0.00831567 0.00982234 0.01391234 0.01405333
 0.01454801 0.01663779 0.01722644 0.01724869 0.02118052 0.0229155
 0.03107507 0.10279346 0.11432321 0.1687874  0.16907775 0.20711204]
'''

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    selection_model = XGBClassifier(n_estimators=300, learning_rate=0.1, n_jobs=-1)
    selection_model.fit(select_x_train, y_train, verbose=False, eval_metric=['logloss','auc'],
                eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                early_stopping_rounds=20)
    
    y_pred = selection_model.predict(select_x_test)

    acc = accuracy_score(y_test, y_pred)

    print('Thresh=%.3f, n=%d, acc: %.2f%%' %(thresh, select_x_train.shape[1], acc*100.0))
    model.save_model('./model/xgb_save/cancer_n=%d_acc=%.3f.model' %(select_x_train.shape[1], acc))

# Thresh=0.000, n=30, acc: 96.49%
# Thresh=0.001, n=29, acc: 96.49%
# Thresh=0.001, n=28, acc: 96.49%
# Thresh=0.002, n=27, acc: 96.49%
# Thresh=0.002, n=26, acc: 96.49%
# Thresh=0.002, n=25, acc: 96.49%
# Thresh=0.003, n=24, acc: 97.37%
# Thresh=0.004, n=23, acc: 96.49%
# Thresh=0.004, n=22, acc: 96.49%
# Thresh=0.005, n=21, acc: 96.49%
# Thresh=0.005, n=20, acc: 96.49%
# Thresh=0.006, n=19, acc: 96.49%
# Thresh=0.007, n=18, acc: 97.37%
# Thresh=0.007, n=17, acc: 96.49%
# Thresh=0.008, n=16, acc: 96.49%
# Thresh=0.010, n=15, acc: 96.49%
# Thresh=0.014, n=14, acc: 97.37%
# Thresh=0.014, n=13, acc: 97.37%
# Thresh=0.015, n=12, acc: 98.25%
# Thresh=0.017, n=11, acc: 96.49%
# Thresh=0.017, n=10, acc: 97.37%
# Thresh=0.017, n=9, acc: 96.49%
# Thresh=0.021, n=8, acc: 96.49%
# Thresh=0.023, n=7, acc: 96.49%
# Thresh=0.031, n=6, acc: 96.49%
# Thresh=0.103, n=5, acc: 95.61%
# Thresh=0.114, n=4, acc: 92.11%
# Thresh=0.169, n=3, acc: 90.35%
# Thresh=0.169, n=2, acc: 88.60%
# Thresh=0.207, n=1, acc: 88.60%