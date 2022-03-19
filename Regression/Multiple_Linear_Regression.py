import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#Reading dataset and spliting into x and y values
dataset = pd.read_csv('datasets/50_Startups.csv')
x = dataset.drop('Profit', axis=1)
y = dataset['Profit']

#Chonverting categorical data into numerical data
states = pd.get_dummies(x, drop_first=True)
x = x.drop('State', axis=1)
x = pd.concat([x,states], axis=1)

#Splitting the dataset into training and testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=1/3)

#Using Linear Regression from sklearn to predict training and testing data
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_train_pred = regressor.predict(x_train)
y_test_pred = regressor.predict(x_test)

print("Test Predictions: ",y_test_pred)

#Predicting the accuracy score
print("R2 score: ", r2_score(y_test, y_test_pred))
print("Mean Squared Error: ", mean_squared_error(y_test, y_test_pred))
