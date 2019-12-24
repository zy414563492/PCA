import pickle
import numpy as np
from matplotlib import pyplot as plt


data = pickle.load(open("ytc_py.pkl", 'rb'))
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']


print("Number of training image-sets: ", len(X_train))
print("Number of testing image-sets: ", len(X_test))
print("Feature dimension of each image: ", X_train[0].shape, y_train)

y_train_new = y_train[::3]
y_test_new = y_test[::6]
print("y_train_new:", len(y_train_new), y_train_new)
print("y_test_new:", len(y_test_new), y_test_new)


def get_pca_space(X, set_size):
    pcaSpace = []
    for id in range(len(X) // set_size):
        print("id", id)
        useX = []
        for setId in range(set_size):
            if setId == 0:
                useX = X[id * set_size]
            else:
                useX = np.hstack((useX, X[id * set_size + setId]))

        # 这里用上次方法找出第一成分的PCA向量，最后返回这些向量构成的矩阵
        meanX = np.array([np.mean(useX.T[:, i]) for i in range(400)])
        normX = useX.T - meanX
        covX = np.cov(normX.T)
        featValue, featVec = np.linalg.eig(covX)
        featVec = np.real(featVec)
        index = np.argsort(-featValue)
        pcaMat = np.array(featVec.T[index[:10]]).T
        pcaSpace.append(pcaMat)

    return pcaSpace


def get_test_norm(X):
    testSpace = []
    for id in range(len(X) // 6):
        print("id", id)
        useX = []
        for setId in range(6):
            if setId == 0:
                useX = X[id * 6]
            else:
                useX = np.hstack((useX, X[id * 6 + setId]))
        testSpace.append(useX)

    return testSpace


# 得到训练集和测试集的PCA向量矩阵
trainSpace = get_pca_space(X_train, 3)
print("trainSpace: ", len(trainSpace), trainSpace[0].shape)
# testSpace = get_test_norm(X_test)
testSpace = get_pca_space(X_test, 6)
print("testSpace: ", len(testSpace))


# 计算训练集和测试集空间的最小距离
min_idx = [0 for _ in range(len(testSpace))]
for i in range(len(testSpace)):
    print(i)
    score = np.inf
    for j in range(len(trainSpace)):
        dist = [np.inf for _ in range(10)]
        for k in range(10):
            for m in range(10):
                dist_cur = np.linalg.norm(trainSpace[j].T[k] - testSpace[i].T[m])
                if dist_cur < dist[k]:
                    dist[k] = dist_cur
        dist_avg = np.mean(dist)

        if dist_avg < score:
            score = dist_avg
            min_idx[i] = j
    # print("score:{} min_idx:{}".format(score, min_idx[i]))

print("max_idx:", len(min_idx), min_idx)

# 根据上一步的index找到其对应的训练集的label
predict = []
for idx in min_idx:
    predict.append(y_train_new[idx])
print("predict:", predict)
print("y_test_new:", y_test_new)

# 上一步的预测结果和测试集自带的label进行比较，计算正确率
accuracy = []
for i in range(len(testSpace)):
    accuracy.append(predict[i] == y_test_new[i])
print("accuracy:", np.sum(accuracy) / len(testSpace))


