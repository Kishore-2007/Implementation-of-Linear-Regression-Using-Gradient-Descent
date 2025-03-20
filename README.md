# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```python
/*
Program to implement the linear regression using gradient descent.
Developed by: KISHORE.S 
RegisterNumber: 212224230130 
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
## Value of X:

![Screenshot 2025-03-20 083923](https://github.com/user-attachments/assets/dc10b322-84c2-429e-95c4-69dcc5833a48)

![Screenshot 2025-03-20 084007](https://github.com/user-attachments/assets/75d4b73b-00b7-43aa-8905-c3ac21ca187d)

## Value of X1_Scaled:

![Screenshot 2025-03-20 084052](https://github.com/user-attachments/assets/72828f9a-b646-4701-a28d-322d76e2f4f1)

![Screenshot 2025-03-20 084118](https://github.com/user-attachments/assets/d66d67dd-dc34-4b52-b7f4-c9cc5c8c1390)

## Predicted Values:

![Screenshot 2025-03-20 084126](https://github.com/user-attachments/assets/cf1d31c6-55b9-4391-94f0-c6b68fa22cfe)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
