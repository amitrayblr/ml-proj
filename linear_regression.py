import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt

data = pd.read_csv("file location")
print ("The shape of the data is", data.shape)
data.head()

X = data["X Label"].values
Y = data["Y Label"].values

#Plotting scatter plot for data
plt.scatter(X, Y, marker = 'x', c = '#FF6347', label = "Scatter Plot")
plt.show()

#Defining mean function
def average(lst):
    return(sum(lst)/len(lst))

#Finding mean of the data
mean_x = average(X)
mean_y = average(Y)

#Finding the coefficients b1 and b0
numerator = 0
denominator = 0
for i in range(len(X)):
    numerator += (X[i] - mean_x)*(Y[i] - mean_y)
    denominator += (X[i] - mean_x)**2
    b1 = numerator/denominator
    b0 = mean_y - (b1*mean_x)
print("The coefficients are: ", b1, b0)

max_x = nm.max(X) + 100
min_x = nm.min(X) - 100

#Line Values
x = nm.linspace(min_x, max_x, 1000)
y = b0 + b1*x

#Plotting scatter plot for demo
plt.scatter(X, Y, marker = 'x', c = '#FF6347', label = "Scatter Plot")

#Plotting best fit line
plt.plot(x, y, c = '#008080', label = "Best Fit Line")
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.legend()
plt.show()

#Calculating Error
error = 0
for i in range(len(X)):
    pred = b0 + b1 * X[i]
    error += (Y[i] - pred)**2
error = nm.sqrt(error/len(X))
print("Calculate error is: ", error)

#Evaluation of Model
totsum = 0
resisum = 0
for i in range(len(X)):
    pred = b0 + b1 * X[i]
    totsum += (Y[i] - mean_y) ** 2
    resisum += (Y[i] - pred) ** 2
coefdet = 1 - (resisum/totsum)
print("The Coefficient of Determination is: ", coefdet)

hp = int(input("Enter your input"))
mpg = b0 + b1 * hp
print("The expected output is ", mpg)
