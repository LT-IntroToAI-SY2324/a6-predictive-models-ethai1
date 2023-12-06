# Part 3 - Multivariable Linear Regression Writeup

After completing `a6_part3.py` answer the following questions

## Questions to answer

1. What is the R Squared coefficient for your model? What does that mean about the model in relation to your data?
The R_squared value is 0.85. This shows me that the miles(000) is well correlated with the price of the car.

2. Is your model accurate? Why or why not?
The model is accurate because the R_squared value is high which shows that the data is well correlated with the two variables. In real life, the car's value depreciates as it ages and gets more miles on it which is exactly reflected in the model.

3. What does the model predict a 10-year-old car with 89000 miles is worth? What about a car that is 20 years old with 150000 miles?
Model predicts a 10 year old car with 89,000 is $8975.88 dollars and the 20 year old car with 150,000 miles on it is worth $2348.53 dollars.

4. You may notice that some of your predicted results are negative. This is occurring when the value of age and the mileage of the car are very high. Why do you think this is happening?
It happens since the coefficients are negative and since the independent variables (miles(000) and age) are high, the predicted price will be negative as it goes below the x-axis, which doesn't make sense in reality but does mathematically. 
