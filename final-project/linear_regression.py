import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("final-project/real_estate_data.csv")
x = data["Assessed Value"].values
y = data["Sale Amount"].values

x = x.reshape(-1, 1)

model = linear_model.LinearRegression().fit(x, y)
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)

# Not correlated at all
r_squared = model.score(x, y)

print(f"Linear Equation is: y = {coef}x + {intercept}")
print(f"R squared value is: {r_squared}")

plt.scatter(x, y)
plt.xlabel("Assessed Value")
plt.ylabel("Sale Amount")
plt.show()