import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("part4-classification/suv_data.csv")
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)

x = data[["Age", "EstimatedSalary", "Gender"]].values
y = data["Purchased"].values

# Step 1: Print the values for x and y
print(x, y)

# Step 2: Standardize the data using StandardScaler, 
scale = StandardScaler().fit(x)

# Step 3: Transform the data
x = scale.transform(x)

# Step 4: Split the data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# Step 5: Fit the data

# Step 6: Create a LogsiticRegression object and fit the data
model = linear_model.LogisticRegression().fit(xtrain, ytrain)

# Step 7: Print the score to see the accuracy of the model
print(f"Accuracy: ", + model.score(xtest, ytest))

test = model.predict([[1, 34, 56000]])
print(f"34 year olf female decided to: {test}")

# Step 8: Print out the actual ytest values and predicted y values
# based on the xtest data
predict = model.predict(xtest)
x = x.reshape(-1, 2)
for index in range(len(xtest)):
    x_coordinates = xtest[index]
    y_prediction = predict[index]

    if y_prediction == 0:
        y_prediction = "Bought"
    else:
        y_prediction = "Not bought"

    actual_value = ytest[index]

    if actual_value == 0:
        actual_value = "Bought"
    else:
        actual_value = "Not bought"

    print(f"Predicted Decision: {y_prediction}, Actual Decision: {actual_value}")