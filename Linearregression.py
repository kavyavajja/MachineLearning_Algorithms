import sys
import csv
import numpy as np
from numpy import genfromtxt

y_pred = 0
SSE = 0
SSE1 = 0
iteration = 0
threshold = float(sys.argv[6])
learning_rate = float(sys.argv[4])
my_data = genfromtxt ( sys.argv[2], delimiter=',' )
sample_size = len ( my_data )
weights = [0.0] * len ( my_data[3] )
x_given = my_data[:, :-1]
x0 = np.array ( [1] * len ( my_data ) )
X = np.column_stack ( (x0, x_given) )
y_actual = my_data[:, -1]
err = y_actual[0]
while True:
        y_pred = np.dot ( X, weights )
        error = (y_actual - y_pred)
        SSE1 = sum ( error * error )
        solution = weights[:]
        solution.insert ( 0, iteration )
        solution.extend ( [SSE1] )
        rounded = [round ( num, 6 ) for num in solution]
        rounded_str =[str ( x ) for x in rounded]
        print ( ",".join(rounded_str) )
        if (abs ( SSE1 - SSE ) <= threshold):
            break
        for i in range ( len ( weights ) ):
            gradient = np.dot ( X[:, i], error )
            update_value = learning_rate * gradient
            weights[i] += update_value
        SSE = SSE1
        iteration += 1

