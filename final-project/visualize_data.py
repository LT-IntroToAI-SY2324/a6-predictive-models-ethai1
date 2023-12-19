import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("final-project/real_estate_data.csv")
data["Residential Type"].replace(["Single Family", "Two Family", "Condo", "Nan"], [0, 1, 2, 3], inplace=True)
data["Property Type"].replace(["Residential", "Commercial", "Vacant Land"], [0, 1, 2], inplace=True)
x = data[["Assessed Value", "Sales Ratio"]].values
y = data["Sale Amount"].values

scaler = StandardScaler().fit(x, y)

x_1 = data["Assessed Value"]
x_2 = data["Sales Ratio"]
x_1 = scaler.transform(x_1)
x_2 = scaler.transform(x_2)

x_1 = x_1.reshape(-1, 2)
x_2 = x_2.reshape(-1, 2)

fig, graph = plt.subplots(2)
graph[0].scatter(x_1, y)
graph[0].set_xlabel("Assessed Value")
graph[0].set_ylabel("Sales Amount")

graph[1].scatter(x_2, y)
graph[1].set_xlabel("Sales Ratio")
graph[1].set_ylabel("Sales Amount")

plt.tight_layout()
plt.show()