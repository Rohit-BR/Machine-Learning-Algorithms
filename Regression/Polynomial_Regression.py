import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Reading dataset and spliting into x and y values
dataset = pd.read_csv('datasets/Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

lin_reg=LinearRegression()
lin_reg.fit(x,y)

#fitting the polynomial regression model to the dataset
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(x)
poly_reg.fit(X_poly,y)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualising the pollynomial regression model results
X_grid=np.arange(min(x),max(x),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.savefig("outputs/PR_Output_1.png")
plt.show()

#predicting the result of polynomial regression.
test_data = float(input("Enter level of experience: "))
test_pred = lin_reg2.predict(poly_reg.fit_transform(np.array([ [test_data] ]) ))
print("Predicted Salary:",test_pred[0])

