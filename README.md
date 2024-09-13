# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load California housing data, select features and targets, and split into training and testing sets.
2. Scale both X (features) and Y (targets) using StandardScaler.
3. Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.
4. Predict on test data, inverse transform the results, and calculate the mean squared error.

## Program and Output:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: swathi.s
RegisterNumber:   212223040219.
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
```
```
# load the California Housing dataset 
data = fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())

````
![image](https://github.com/user-attachments/assets/257cb9bc-1ac2-494d-a1ce-e991567d4b18)
```
df.info()
```
![image](https://github.com/user-attachments/assets/3e49893a-9b88-4d8a-9f68-ba9b4f262953)
```
X=df.drop(columns=['AveOccup','target'])
```
```
X.info()
```
![image](https://github.com/user-attachments/assets/26e85178-9425-4c27-87bd-842e1cb986ca)
```
Y=df[['AveOccup','target']]
```
```
Y.info()
```
![image](https://github.com/user-attachments/assets/2089034e-db5b-466a-9ca0-de5a6ca07e26)
```
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
```
```
X.head()
```
![image](https://github.com/user-attachments/assets/2d4ec8c2-6d72-440d-bf4b-0ed5e6de37cc)
```
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
```
```
print(X_train)
```
![image](https://github.com/user-attachments/assets/1b2c6b95-6898-4a4f-a4d9-4f432c54307c)
```
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)
Y_pred = multi_output_sgd.predict(X_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)
Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
```
![image](https://github.com/user-attachments/assets/b4e45168-7764-4ab3-8f4f-bd7e2668311e)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
