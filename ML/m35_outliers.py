import numpy as np

def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    print("1사분위 : ", quartile_1)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))

# a = np.array([1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100])

# b = outliers(a)
# print("이상치의 위치 : ", b)


# 실습 : 행렬을 입력해서 컬럼별로 이상치 발견하는 함수를 구현하시오
# 파일명 : m36_outliers2.py