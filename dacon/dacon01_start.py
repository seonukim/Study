import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('./data/dacon/comp1/train.csv',
                    header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv',
                   header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv',
                         header = 0, index_col = 0)

print('train.shape : ', train.shape)                # (10000, 75) : x_train, test
print('test.shape : ', test.shape)                  # (10000, 71) : x_predict
print('submission.shape : ', submission.shape)      # (10000, 4)  : y_predict

# Null 확인
print(train.isnull().sum())

# 보간법 - 선형 보간법
train = train.interpolate()         # predict 값으로 채워넣어주는 값이다
print(train.isnull().sum())


