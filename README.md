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
7.Model Evaluation 
8.End

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: swathi.s
RegisterNumber:  212223040219.
*/
```
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
```
![image](https://github.com/user-attachments/assets/abe42a9d-224d-4cea-a97f-f928db09c588)

df.info()
![image](https://github.com/user-attachments/assets/931d2f27-6c39-474f-bb7c-e9f9fc535452)

X=df.drop(columns=['AveOccup','target'])
X.info()
![image](https://github.com/user-attachments/assets/2b7d975e-fa44-4850-9194-a12938ed88a3)

Y=df[['AveOccup','target']]
Y.info()




## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
