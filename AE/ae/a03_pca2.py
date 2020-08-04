import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

x = dataset.data
y = dataset.target

print(x.shape) # (442, 10)
print(y.shape) # (442,)

# pca = PCA(n_components=5)
# x2 = pca.fit_transform(x)
# pca_evr = pca.explained_variance_ratio_  # 압축한 컬럼별 중요비율
# print(pca_evr)
# print(sum(pca_evr))  # 0.83 -> pca해서 중요한 0.17은 소실했다

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

n_components = np.argmax(cumsum >= 0.94)+1  # argmax가 어찌 돌아가는지는 찍어봐야 알듯.. 제일 큰거 아니었어..?
print(cumsum>=0.94)
print(n_components)




