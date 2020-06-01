import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dropout
from keras.layers import Dense, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

scaler = RobustScaler()
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
pca = PCA(n_component = 8)

