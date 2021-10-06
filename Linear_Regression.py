import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Reading dataset and spliting into x and y values
dataset = pd.read_csv('datasets/Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting the dataset into training and testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=1/3)

#Using Linear Regression from sklearn to predict training and testing data
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_train_pred = regressor.predict(x_train)
y_test_pred = regressor.predict(x_test)

#Plotting Training Data
plt.scatter(x_train, y_train, color = 'blue')
plt.plot(x_train, y_train_pred, color = 'red')
plt.savefig("outputs/Graph1.png")

#Plotting Testing Data
plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_test, y_test_pred, color = 'red')
plt.savefig("outputs/Graph2.png")