#xgbooster에도 evaluation이 있다. 
from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score

# datasets = load_boston()
# x = dataset.data
# y = dataset.target

datasets = load_breast_cancer()


x, y = load_breast_cancer(return_X_y=True)# 이렇게 적어줘도됨

print(x.shape) #(569, 30)
print(y.shape) #(569,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
                  shuffle = True, random_state = 66)

model = XGBClassifier(n_estimators = 400, learning_rate = 0.1)

model.fit(x_train, y_train, verbose=True, eval_metric="error",
 eval_set=[(x_train, y_train), (x_test, y_test)])
#rmse, mae, logloss, error, auc
#error은 회기 모델 지표가 아니다


# result = model.evals_result()
# print("eval's results :", result)

# r2 = r2_score(y_predict, y_test)
# print("r2 Score : %.2f%%" %(r2 * 100.0))
# print("r2 :", r2)
y_predict = model.predict(x_test)
acc = accuracy_score(y_predict, y_test)
print("acc : ", acc)
###################################################################################


# import pickle#파이썬에서 제공하는 피클
# pickle.dump(model, open("./model/xgb_save/cancer.pickle.data", "wb"))
# print("SAVED!!!!")
#피클과 잡립을 비교해보아라 둘다 저장라는 방법임
# from joblib import dump, load

# import joblib
# joblib.dump(model, "./model/xgb_save/cancer.joblib.data")
# print("SAVED!!!!")
model.save_model("./model/xgb_save/cancer.xgb.model")
print("SAVED!!!!")
 


# 불러오기 
# model2= pickle.load(open("./model/xgb_save/cancer.pickle.data", "rb"))

# model2= joblib.load("./model/xgb_save/cancer.joblib.data")
model2= XGBClassifier()
model2.load_model("./model/xgb_save/cancer.xgb.model")

print("LOADED!!!!불러왔다. ")
y_predict = model.predict(x_test)
acc = accuracy_score(y_predict, y_test)
print("acc : ", acc)
#저장을하고 다시 불러온거에서 두개의 애큐러시가 동일한지를 확인해주는 과정을 거치면 두개다 동일한거에서 저장하고 다시 불러오기 한것임을 알 수 있다. 