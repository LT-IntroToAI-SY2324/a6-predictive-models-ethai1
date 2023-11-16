import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Use reshape to turn the x values into 2D arrays:
x = x.reshape(-1,1)

# Create the model
model = LinearRegression().fit(x, y)

# Find the coefficient, bias, and r squared values. 
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)

# Each should be a float and rounded to two decimal places. 


# Print out the linear equation and r squared value
print(f"Linear Equation is: y = {coef}x + {intercept}")
print(f"R squared value is: {r_squared}")

# Predict the the blood pressure of someone who is 43 years old.
predict_age = 43
prediction = model.predict([[predict_age]])
# Print out the prediction
print(prediction)
# Create the model in matplotlib and include the line of best fit

plt.figure(figsize=(6, 4))

plt.scatter(x, y)
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure by Age")

plt.show()