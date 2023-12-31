import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles(000)","age"]].values
y = data["Price"].values

#split the data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
print(xtest)

#create linear regression model
model = LinearRegression().fit(xtrain, ytrain)

#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 

coefficent = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(x, y), 2)

print(f"Miles(000) coefficient is: {coefficent[0]}")
print(f"Age coefficient is: {coefficent[1]}")
print(f"Intercept is: {intercept}")
print(f"R_squared value is: {r_squared}")
print(f"Model equation is y = {coefficent[0]}x1 + {coefficent[1]}x2 + {intercept}")

# first attempt, not using predict function
# test1 = coefficent[0] * 89 + coefficent[1] * 10 + intercept
# test2 = coefficent[0] * 150 + coefficent[1] * 20 + intercept
# print(f"10 year old car, 89,000 miles: ${test1}")
# print(f"20 year old car, 150,000 miles: ${test2}")

prediction = model.predict(xtest)

test3 = model.predict([[89, 10]])
test4 = model.predict([[150, 20]])
print("Tests 3 and 4", test3, test4)

prediction = np.around(prediction, 2)

#Loop through the data and print out the predicted prices and the 
#actual prices
print("***************")
print("Testing Results")

for num in range(len(xtest)):
    actual_value = ytest[num]
    predicted_y = prediction[num]
    x_coordinate = xtest[num]

    print(f"The miles(000) is: {x_coordinate[0]}, Age is: {x_coordinate[1]}, Predicted Price: ${predicted_y}, Actual Price: ${actual_value}")

