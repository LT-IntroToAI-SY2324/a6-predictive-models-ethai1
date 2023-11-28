import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

data = pd.read_csv("data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Create your training and testing datasets:
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# Use reshape to turn the x values into 2D arrays:
xtrain = xtrain.reshape(-1,1)

# Create the model
model = LinearRegression().fit(xtrain, ytrain)

# Find the coefficient, bias, and r squared values. 
coefficient = round(float(model.coef_))
bias = round(float(model.intercept_))
r_squared = model.score(xtrain, ytrain)

# Each should be a float and rounded to two decimal places. 


# Print out the linear equation and r squared value:
print(f"The equation is y = {coefficient}x + {bias}")
print(f"r_squared value is {r_squared}")

'''
**********TEST THE MODEL**********
'''
# reshape the xtest data into a 2D array
xtest = xtest.reshape(-1, 1)

# get the predicted y values for the xtest values - returns an array of the results
predictions = model.predict(xtest)

# round the value in the np array to 2 decimal places
predictions = np.around(predictions, 2)

# Test the model by looping through all of the values in the xtest dataset
print("\nTesting Linear Model with Testing Data:")

for num in range(len(xtest)):
    x_prediction = xtest[num]
    y_prediction = ytest[num]
    
'''
**********CREATE A VISUAL OF THE RESULTS**********
'''

# plt.figure(6, 4)

# plt.scatter(x, y)
