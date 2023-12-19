import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("final-project/real_estate_data.csv")
data["Residential Type"].replace(["Single Family", "Two Family", "Condo", "Nan", "Continuous"], [50000, 75000, 150000, 0, 0], inplace=True)
data["Property Type"].replace(["Residential", "Commercial", "Vacant Land", "Continuous"], [50000, 75000, 25000, 0], inplace=True)
x = data["Assessed Value", "Residential Type", "Property Type"].values
y = data["Sale Amount"]

# scaler = StandardScaler().fit(x)
# x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = linear_model.LogisticRegression().fit(x_train, y_train)

#Prints the accuracy and predictions of the model
print("Accuracy:", model.score(x_test, y_test))
print("*************")
print("Testing Results:")
print("")
print(y_test)
for index in range(len(x_test)):
    x = x_test[index]
    x = x.reshape(-1, 4)
    y_pred = int(model.predict(x))
    
    actual = y_test[index]

    print("Predicted Price: " + y_pred + " Actual Price: " + actual)

