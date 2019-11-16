#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd



df = pd.read_csv('exoTest.csv')
rowX = df.values
print("rowX:", rowX.shape)

cub_len = 56
n_feat = cub_len ** 2
print(n_feat)
usedX = rowX[:, 1:1 + n_feat]
print("usedX:", usedX.shape)
print(usedX)


meanX = np.array([np.mean(usedX[:, i]) for i in range(n_feat)])
print("meanX:\n", meanX)

# Normalization
normX = usedX - meanX
print("normX.shape:", normX.shape)
print(normX)

# Calculating the covariance matrix
covX = np.cov(normX.T)
print("covX.shape:", covX.shape)
print(covX)

# Find eigenvalues and eigenvectors
featValue, featVec = np.linalg.eig(covX)
print("featVec: ", featVec.shape)
print(featVec)
featVec = np.real(featVec)

# Sort by feature values
index = np.argsort(-featValue)

n_pri = 9
priVec = np.array(featVec.T[index[:n_pri]]).T
print("priVec: ", priVec.shape)


plt.figure()
plt.suptitle('1~9 PCA vectors')
for i in range(n_pri):
    feat_cub = priVec.T[i].reshape((cub_len, cub_len))
    plt.subplot(3, 3, 1 + i)
    plt.axis('off')
    plt.title('{}'.format(1 + i))
    plt.imshow(feat_cub)
plt.show()


finalX = np.dot(normX, priVec)
print("finalX:", finalX.shape)
print(finalX)


coordinate = np.dot(priVec.T, normX.T)
print("coordinate:", coordinate.shape)
print(coordinate)


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.plot(coordinate[0], coordinate[1], coordinate[2], marker="o", linestyle='None')
plt.title('1~3 space')
plt.show()


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.plot(coordinate[3], coordinate[4], coordinate[5], marker="o", linestyle='None')
plt.title('4~6 space')
plt.show()


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.plot(coordinate[6], coordinate[7], coordinate[8], marker="o", linestyle='None')
plt.title('7~9 space')
plt.show()










