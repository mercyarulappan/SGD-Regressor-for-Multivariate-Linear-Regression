# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start Step

2.Data Preparation

3.Hypothesis Definition 

4.Cost Function 

5.Parameter Update Rule

6.Iterative Training 

7.Model Evaluation 8.End

## Program:
```python
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: MERCY A
RegisterNumber:  212223110027
*/

#Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
print(df.tail())

 
df.info()
x=data.data[:,:3]

x=df.drop(columns=['AveOccup','target'])
x.info()

y=df[['AveOccup','target']]
y.info()

x.head()

scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)

print(X_train)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
```

## Output:
![image](https://github.com/user-attachments/assets/6cb31e6d-a610-4df7-bb97-ff65a84fdf98)

![image](https://github.com/user-attachments/assets/b941d97b-e1e2-4aea-b2bb-1d3ba4078033)

![image](https://github.com/user-attachments/assets/8c7e4efe-8ecb-4785-8e2b-01985ff5efce)

![image](https://github.com/user-attachments/assets/355211d7-8fe0-4f90-a5e2-e839dcd9fd59)

![image](https://github.com/user-attachments/assets/cc8208e0-b04a-4350-bb8e-56b7ad0176b9)

![image](https://github.com/user-attachments/assets/1f9d7e17-5a1c-459e-857e-6530f5f7d0df)

![image](https://github.com/user-attachments/assets/dd25ee75-1eaa-4271-96bd-33607b302b18)







## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
