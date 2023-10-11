# importing the required libraries

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('apartments.csv', thousands=' ')
data['rate'] = data.rate.str.contains('Per Month').astype(int)
data["price"] = pd.to_numeric(data["price"])
data.isnull().sum()
data = data.dropna()

# Mapping specific location names to numeric codes for 'location' column:

# pd.value_counts(data['location'])  # getting the unique values
data.loc[data['location'] == 'Kileleshwa, Nairobi','location'] = 1
data.loc[data['location'] == 'Westlands, Nairobi','location'] = 2
data.loc[data['location'] == 'Kilimani, Nairobi','location'] = 3
data.loc[data['location'] == 'Shanzu, Mombasa','location'] = 4
data.loc[data['location'] == 'Kikuyu Town Bus park Kikuyu, Kikuyu, Kikuyu','location'] = 5
data.loc[data['location'] == 'Westlands downtown, Westlands, Nairobi','location'] = 6
data.loc[data['location'] == 'Links Rd Mombasa, Nyali, Mombasa','location'] = 7
data.loc[data['location'] == 'Muthaiga, Nairobi','location'] = 8
data.loc[data['location'] == 'Kileleshwa Nairobi, Kileleshwa, Nairobi','location'] = 9
data.loc[data['location'] == 'Grevillea Grove Spring Valley, Spring Valley, Nairobi','location'] = 10
data.loc[data['location'] == 'Riverside Dr Nairobi, Riverside, Nairobi','location'] = 11
data.loc[data['location'] == 'Nyali, Mombasa','location'] = 12
data.loc[data['location'] == 'Thika Rd Nairobi, Kahawa Wendani, Nairobi','location'] = 13
data.loc[data['location'] == 'Vihiga road, Kileleshwa, Nairobi','location'] = 14
data.loc[data['location'] == 'Near Valley Arcade, Lavington, Nairobi','location'] = 15
data.loc[data['location'] == 'Off Othaya road, Lavington, Nairobi','location'] = 16
data.loc[data['location'] == 'Jabavu court, Kilimani, Nairobi','location'] = 17
data.head(5)

X = data[["location","bedrooms","bathrooms"]].values
y = data["price"].to_numpy()
m = y.size

# Handling rates: Replace zero rates with 30 and calculate 'Y' based on 'y'
Rate = data['rate'].to_numpy()
Rate[Rate == 0] = 30
Y = y*Rate
Y
#price = np.stack([y, Rate], axis=1)

def plotData(X,Y):
    plt.plot(X, Y, 'ro', ms=10, mec='k')
    plt.ylabel('price of house')
    plt.xlabel('Location,Number of bedrooms and bathrooms')


plotData(X,Y)


X = np.concatenate([np.ones((m,1)), X], axis=1)

### Using the method of least squares

theta = np.zeros(X.shape[1])
theta = np.linalg.inv(X.T@X)@X.T@Y
theta
 

predicted_price = np.dot([1,11,3,3],theta)
predicted_price


### Using the batch gradient descent algorithm

def computeCost(X,Y,theta):
    J = 0
    J = (1/(2*m))*np.sum((np.dot(X,theta)-y)**2)
    return J
    


def gradientDescent(X, Y, theta, alpha, num_iters):
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        h = np.dot(X,theta)
        theta = theta - alpha*(1/m)*np.dot(X.T,(h-y))
        J_history.append(computeCost(X, Y, theta))
    return theta, J_history


theta = np.zeros(4)
iterations = 5000
alpha  = 0.01
theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
print(theta)
print(J_history)

predicted_price = np.dot([1,11,3,3],theta)
predicted_price

ss_t = 0
ss_r = 0
mean_y = np.mean(Y)
for i in range(m):
    y_pred = np.dot(X[i],theta)
    ss_t += (Y[i] - mean_y)**2
    ss_r += (Y[i] - y_pred)**2
r2 = 1 - (ss_r/ss_t)
print(r2)
