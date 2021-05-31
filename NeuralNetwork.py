#   A Very Simple Neural Network in Python 3 with Numpy, Part 2

import sys

import numpy as np
from numpy import genfromtxt
from math import exp

my_data = genfromtxt (sys.argv[2], delimiter=',',dtype=float)
sample_size = len ( my_data )
x_given  = my_data[:, :-1]
target =  my_data[:,-1]
X0 = np.ones((sample_size,1))
X_inp = np.hstack((X0,x_given))
Wh = np.array([[0.2, -0.3, 0.4], [-0.5, -0.1, -0.4], [0.3, 0.2, 0.1]])
Wz = np.array([-0.1, 0.1, 0.3, -0.4])
lr = float(sys.argv[4])
iterations = int(sys.argv[6])
hiddenlayers =3
inputs =3

def sigmoid(x):
    return 1/(1+np.exp(-x))
def rounded(X):
    rounded = ['{0:.{1}f}'.format ( num, 5) for num in X]
    rounded_str = [str ( x ) for x in rounded]
    print ( ",".join ( rounded_str ),end=",")
def rounded_last(X):
    rounded = ['{0:.{1}f}'.format ( num, 5) for num in X]
    rounded_str = [str ( x ) for x in rounded]
    print ( ",".join ( rounded_str ))
print("-,"*11,end=",")
for j in range ( hiddenlayers ):
    for k in range ( hiddenlayers ):
        print ( Wh[j, k], end="," )
rounded_str = [str ( x ) for x in Wz]
print ( ",".join ( rounded_str ))
for i in range(iterations):
    for i in range(sample_size):
        XWh= np.empty(hiddenlayers, dtype=float)
        z=np.empty(hiddenlayers, dtype=float)
        delta_z=np.empty(hiddenlayers, dtype=float)
        delta_y =0
        rounded ( x_given[i] )
        # feedforward step1
        for j in range(hiddenlayers):
            XWh[j] = np.dot(X_inp[i],Wh[j])
            z[j] = sigmoid (XWh[j])
        hiden_X=np.hstack((1,z))
        XWo=np.dot(hiden_X,Wz)
        y= sigmoid(XWo)
        delta_y = y*(1-y)*(target[i]-y)
        rounded(z)
        print ( '{0:.{1}f}'.format( y, 5 ), end="," )
        print(target[i],end=",")
        for j in range ( hiddenlayers ):
            delta_z[j]=z[j]*(1-z[j])*Wz[j+1]*delta_y
        rounded ( delta_z )
        print ( round ( delta_y, 5 ), end="," )
        for j in range ( hiddenlayers ):
            for k in range ( hiddenlayers ):
                 Wh[j,k]=Wh[j,k]+lr*delta_z[j]*X_inp[i,k]
                 print('{0:.{1}f}'.format(Wh[j,k],5),end=",")
        Wz=Wz+lr*delta_y*hiden_X
        rounded_last(Wz)
