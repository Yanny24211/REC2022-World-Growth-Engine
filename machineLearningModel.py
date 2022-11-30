import sklearn.model_selection
import numpy as np
import pandas as pd
from sklearn import linear_model
import pickle

data = pd.read_csv("world-growth-chosen.csv", sep=",")

# print(data.head())

data = data[["Population Growth Rate (percentage)", "Rate of "
                                                    "Natural Change "
                                                    "(per 1,"
                                                    "000 "
                                                    "population)",
             "Net Migration Rate (per 1,000 population)", "Location code", "Year"]]

print(data.head())
predict = "Population Growth Rate (percentage)"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.01)

# Start of training process
linear = linear_model.LinearRegression()

linear.fit(xTrain, yTrain)
accuracy = linear.score(xTest, yTest)
print(accuracy)

with open("bestWorldGrowthModel.pickle", "wb") as f:
    pickle.dump(linear, f)

stdInPickle = open("bestWorldGrowthModel.pickle", "rb")
linear = pickle.load(stdInPickle)


# Do predictions on the test data that we
# did not train our model on
modelPredictions = linear.predict(xTest)

# Loop through the testing data and print out the results for each row
for x in range(len(modelPredictions)):
    print(modelPredictions[x], xTest[x], yTest[x])



