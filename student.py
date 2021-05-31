import collections
import math
import sys

import numpy as np
from numpy import genfromtxt
from math import exp
from math import sqrt
import string

dataset = genfromtxt(sys.argv[2], delimiter=',', dtype=str)
size =len(dataset)
DT = np.array(dataset[:,[1,2]])
X_train=DT.astype(np.float)
y_train = np.array(dataset[:,0])
prediction=[]

for i in range(0,y_train.shape[0]):
    if y_train[i] == "A":
        y_train[i] = 0
    else:
        y_train[i] = 1

def pre_prob(y):
    y_dict = collections.Counter(y)
    pre_probab = np.ones(2)
    for i in range(0, 2):
        pre_probab[i] = y_dict[i]/y.shape[0]
    return pre_probab

def mean_var(X, y):
    n_features = X.shape[1]
    m = np.ones((2, n_features))
    v = np.ones((2, n_features))
    n_0 = np.bincount(y)[np.nonzero(np.bincount(y))[0]][0]
    x0 = np.ones((n_0, n_features))
    x1 = np.ones((X.shape[0] - n_0, n_features))
    k = 0
    for i in range(0, X.shape[0]):
        if y[i] == 0:
            x0[k] = X[i]
            k = k + 1
    k = 0
    for i in range(0, X.shape[0]):
        if y[i] == 1:
            x1[k] = X[i]
            k = k + 1

    for j in range(0, n_features):
        m[0][j] = np.mean(x0.T[j])
        v[0][j] = np.var(x0.T[j]) * (n_0 / (n_0 - 1))
        m[1][j] = np.mean(x1.T[j])
        v[1][j] = np.var(x1.T[j]) * ((X.shape[0] - n_0) / ((X.shape[0]
                                                            - n_0) - 1))
    return m, v  # mean and variance
def prob_feature_class(m, v, x):
    n_features = m.shape[1]
    pfc = np.ones(2)

    for i in range(0, 2):
        product = 1
        for j in range(0, n_features):
            product = product * (1/sqrt(2*3.14*v[i][j])) * exp(-0.5
                                 * pow((x[j] - m[i][j]),2)/v[i][j])
        pfc[i] = product
    return pfc
def GNB(X, y, x):
    m, v = mean_var(X, y)
    for i in range(0,size):
        pfc = prob_feature_class(m, v, x[i])
        pre_probab = pre_prob(y)
        pcf = np.ones(2)
        total_prob = 0
        for i in range(0, 2):
            total_prob = total_prob + (pfc[i] * pre_probab[i])
        for i in range(0, 2):
            pcf[i] = (pfc[i] * pre_probab[i])/total_prob
        prediction.append(int(pcf.argmax()))
    return m, v, pre_probab, pfc, pcf

m, v, pre_probab, pfc, pcf = GNB(X_train, y_train.astype(int), X_train)
count =0
n_features = m.shape[1]
for j in range(0, n_features):
    for i in range(2):
      print(m[j,i],end=",")
      print(v[j,i],end=",")
    print(pre_probab[count])
    count+=1

count=0
for i in range(size):
    if(int(y_train[i])!=prediction[i]):
         count+=1
print(count)