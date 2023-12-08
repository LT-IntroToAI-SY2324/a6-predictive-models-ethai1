# Part 4 - Classification Writeup

After completing `a6_part4.py` answer the following questions

## Questions to answer

1. Comment out the StandardScaler and re-run your test. How accurate is the model? Why is that?
It was originally 0.85 but then dropped to 0.4875 which makes the model super inaccurate as 0.4875 is a very low value and a decent model would have to be at least 0.75 to be considered at least somewhat useful.

2. How accurate is the model with the StandardScaler? Is this model accurate enough for the given use case? Explain.
Since the model is 0.85 with StandardScaler, the model is useful enough as it's above the 0.75 threshold for being considered a good model.

3. Looking at the predicted and actual results, how did the model do? Was there a pattern to the inputs that the model was incorrect about?
Looking at the results, it seems the model is very accurate with only around a dozen mistakes and from what I could tell, there wasn't a sensible pattern in the incorrect results.

4. Would a 34 year old Female who makes 56000 a year buy an SUV according to the model? Remember to scale the data before running it through the model.
A 34 year old Female who makes 56,000 dollars a year wouldn't buy a SUV.
