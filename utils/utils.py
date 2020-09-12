import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def classification(X, y, testSize=0.2):
    clf = OneVsRestClassifier(LogisticRegression(max_iter=10000))
    binarizer = MultiLabelBinarizer()
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=testSize, shuffle=True)
    binarizer.fit(y)
    clf.fit(trainX, binarizer.transform(trainY))
    topKList = [len(i) for i in testY]
    probs = np.asarray(clf.predict_proba(np.asarray(testX)))
    for i, k in enumerate(topKList):
        lables = clf.classes_[probs[i, :].argsort()[-k:]].tolist()
        probs[i, :] = 0
        probs[i, lables] = 1
    testY = binarizer.transform(testY)
    return {'micro': f1_score(testY, probs, average='micro'), 'macro': f1_score(testY, probs, average='macro')}


def linkPrediction(embedding, trainGraph, originGraph, precisionK_list):
    similarity = np.dot(embedding, embedding.T).reshape(-1)  # N*N
    sortedInd = np.argsort(similarity)[::-1]
    count, k, precisionK = 0, 0, []
    for index in sortedInd:  # similarity矩阵第i,j个元素
        i = index / trainGraph.nodeSize  # 2405
        j = index % trainGraph.nodeSize
        if i == j or trainGraph.aMatrix.numpy()[int(i), j] == 1:
            continue
        k += 1
        if originGraph.aMatrix.numpy()[int(i), j] == 1:
            count += 1
        if k == precisionK_list[len(precisionK)]:
            precisionK.append(1.0 * count / k)
        if k > precisionK_list[-1] or len(precisionK) >= len(precisionK_list):
            break
    return precisionK
