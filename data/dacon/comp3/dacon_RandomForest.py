import pandas as pd
import numpy as np

train_features = pd.read_csv('./data/dacon/comp3/train_features.csv',
                             encoding = 'utf-8')
train_target = pd.read_csv('./data/dacon/comp3/train_target.csv',
                           index_col = 'id', encoding = 'utf-8')
test_features = pd.read_csv('./data/dacon/comp3/test_features.csv',
                            encoding = 'utf-8')
print(train_features.shape)         # (1050000, 6)
print(train_target.shape)           # (2800, 4)
print(test_features.shape)          # (262500, 6)

print(train_features.describe())