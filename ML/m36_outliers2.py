import numpy as np
import pandas as pd
from collections import Counter

a = np.array([[1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100]])
# print(data.shape)

# target = data[:, 1]
# print(target)


def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 = Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


def search_outliers(data_out):
    outliers = []
    for i in range(data_out.shape[1]):
        data = data_out[:, i]
        quartile_1, quartile_3 = np.percentile(data, [25, 75])
        print(str(i) + "컬럼의 1사분위수, 3사분위수 : ", quartile_1,",", quartile_3)

        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        out = np.where((data > upper_bound) | (data < lower_bound))
        outliers.append(out)
    return print(outliers)


def outliers(data_out):
    out = []
    if str(type(data_out)) == str("<class 'numpy.npdarray'>"):
        for i in range(data_out.shape[1]):
            data = data_out[:, i]
            print(data)

            quartile_1, quartile_3 = np.percentile(data, [25, 75])
            print(str(i) + "번째 컬럼의 1사분위수, 3사분위수 : ", quartile_1, ",", quartile_3)

            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr * 1.5)
            upper_bound = quartile_3 + (iqr * 1.5)
            out_col = np.where((data > upper_bound) | (data < lower_bound))
            print(out_col)
            data = data[out_col]
            print(f'{i + 1}번째 행렬의 이상치 값 : ', data)
            out.append(out_col)
    
    elif str(type(data_out)) == str("<class 'pandas.core.frame.DataFrame'>"):
        for i in data_out.columns:
            data = data_out[i].values
            quartile_1, quartile_3 = np.percentile(data, [25, 75])
            print(str(i) + "번째 컬럼의 1사분위수, 3사분위수 : ", quartile_1, ",", quartile_3)

            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr * 1.5)
            upper_bound = quartile_3 + (iqr * 1.5)
            out_col = data[np.where((data_out > upper_bound) | (data_out < lower_bound))]
            data = data[out_col]
            print(f'{i}의 이상치의 값 : ', data)
            out.append(out_col)
    return out



a = np.array([[1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100],
              [1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100]])
print(a.shape)
print(outliers(a))