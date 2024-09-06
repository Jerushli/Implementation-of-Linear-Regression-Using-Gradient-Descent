# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import necessary libraries such as NumPy, Pandas, Matplotlib, and metrics from sklearn.

2.Load the dataset into a Pandas DataFrame and preview it using head() and tail().

3.Extract the independent variable X and dependent variable Y from the dataset.

4.Initialize the slope m and intercept c to zero. Set the learning rate L and define the number of epochs.

5.In a loop over the number of epochs:

  .Compute the predicted value Y_pred using the formula
  . Calculate the gradients
  
  .Update the parameters m and c using the gradients and learning rate.
  
  .Track and store the error in each epoch.
  
6.Plot the error against the number of epochs to visualize the convergence.
                                                                     
7.Display the final values of m and c, and the error plot.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: JERUSHLIN JOSE JB
RegisterNumber: 212222240039
```

```
import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
## Output:
![Screenshot 2024-09-02 174107](https://github.com/user-attachments/assets/3fe283e3-95b8-4879-a751-6d82e076d3d6)
```
dataset.info()
```
## Output:
![Screenshot 2024-09-02 174214](https://github.com/user-attachments/assets/85a987bb-76da-4798-8135-9a6729c887b1)
```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
## Output:
![Screenshot 2024-09-02 174330](https://github.com/user-attachments/assets/b3b55989-dd23-442f-87d5-dbc67ab7b838)

```
print(X.shape)
print(Y.shape)
```
## Output:
![Screenshot 2024-09-02 174435](https://github.com/user-attachments/assets/7cdd8851-171b-44c8-832a-9677d7e6e08d)
```
m=0
c=0
L=0.0001
epochs=5000
n=float(len(X))
error=[]
for i in range(epochs):
    Y_pred = m*X +c
    D_m=(-2/n)*sum(X *(Y-Y_pred))
    D_c=(-2/n)*sum(Y -Y_pred)
    m=m-L*D_m
    c=c-L*D_c
    error.append(sum(Y-Y_pred)**2)
print(m,c)
type(error)
print(len(error))
```
## Output:
 ![Screenshot 2024-09-02 181348](https://github.com/user-attachments/assets/0ecab5e4-4448-4ecb-9ff4-bec40a8c7459)
 ```
plt.plot(range(0,epochs),error)
```
## Output:
![Screenshot 2024-09-02 181603](https://github.com/user-attachments/assets/74602921-fd42-40b9-ab1b-4834628d0a3d)

 gradient descent is written and verified using python programming.
