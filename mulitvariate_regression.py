import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("File Location")
print ("The shape of the data is", data.shape)
data.head()

#Mean Funciton
def average(lst):
    return(sum(lst)/len(lst))

#Getting the values
mathval = data['X Label'].values
readval = data['Y Label'].values
writeval = data['Z Label'].values

#Plotting the graph
graph = plt.figure()
axis = Axes3D(graph)
axis.scatter(mathval, readval, writeval, marker = "x", color='#0000FF')
plt.show()
 
#Finding the required values
x0 = nm.ones(len(mathval))
X = nm.array([x0, mathval, readval]).T
B = nm.array([0, 0, 0])
Y = nm.array(writeval)
alpha = 0.0001

#Calculating Cost Function
def cost(X, Y, B):
    m = len(X)
    J = nm.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J
print("The cost of the function is: ", cost(X,Y,B))

#Performing Gradient Decent
def gradient_descent(X, Y, B, alpha, iterations):
    cost_his = [0] * iterations
    m = len(Y)
    for i in range(iterations):
        gradient = X.T.dot(X.dot(B) - Y) / m
        B = B - alpha * gradient
        cost_his[i] = cost(X, Y, B)
        
    return(B, cost_his)

new_B, cost_his = gradient_descent(X, Y, B, alpha, 100000)
print(new_B, cost_his[-1])
print("The hypothesis model is ", new_B[0], "+", new_B[1], "*X1", "+", new_B[2], "*X2")

#Finding Root Mean Square Error
def error(Y,pred):
    error = nm.sqrt(sum((Y - pred) ** 2) / len(Y))
    return (error)

#Evaluation of model
def score(Y, pred):
    mean_y = average(Y)
    sumtot = sum((Y - mean_y) ** 2)
    sumresi = sum((Y - pred) ** 2)
    score = 1 - (sumresi/sumtot)
    return(score)

pred = X.dot(new_B)
print("The error is: ", error(Y,pred))
print("The score of the modelled function is: ", score(Y,pred))
