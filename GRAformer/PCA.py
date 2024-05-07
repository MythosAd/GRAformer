import numpy as np

# 收集数据
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 数据标准化
X = (X - np.mean(X,axis=0)) / np.std(X,axis=0)

# 构建协方差矩阵
cov_mat = np.cov(X.T)

# 协方差矩阵特征分解
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

# 得到主成分
# 特征值排序,特征向量矩阵取对应的列
# 特征值排序,特征向量矩阵取对应的列
idx = eigen_vals.argsort()[::-1]
eigen_vecs = eigen_vecs[:,idx]

# 保留2个主成分
eigen_vec_subset = eigen_vecs[:,0:2]

# 转换数据
X_pca = X.dot(eigen_vec_subset)

print(X_pca)