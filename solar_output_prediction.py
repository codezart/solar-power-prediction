import numpy as np
import pandas as pd
import time
import seaborn as seabornInstance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def predict_output():
    dataset = pd.read_csv(file_name)

    # reshape arranges data into a 2d array.
    x = dataset["Time"].values.reshape(-1, 1)
    y = dataset["output"].values.reshape(-1, 1)

    # Data splicing. splits 20% of data for testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # training the model
    regression = LinearRegression()
    regression.fit(x_train, y_train)

    # testing the data and displaying it
    y_pred = regression.predict(x_test)
    df = pd.DataFrame({'Time': x_test.flatten(), 'Actual': y_test.flatten(), 'predicted': y_pred.flatten()})
    print(df)

    # showing how the data looks and the regression line that predicts the data
    plt.scatter(x_test, y_test, color='blue')
    plt.plot(x_test, y_pred, color='red', linewidth=2)
    plt.show()

    y_pred = regression.predict(testing_time_data)
    df = pd.DataFrame({"Time":testing_time_data.flatten(), "Actual": testing_output_data.flatten(),"predicted": y_pred.flatten()})
    df.to_csv("solar_prediction.csv", mode='a')



"""
 .....................................................................................................................
 Start of the program. takes input...

Input is an integer time number of the format hhmm; hours and month in military time.

Condition checks the number and sees if it is before noon or after noon, since the data was divided into two parts 
to get have linearity in the data.

p.s. The data is bell shaped for the whole day :)
"""

file_name = "datasetbnoon.csv"
testing_data = pd.read_csv("testingbnoon.csv")
testing_time_data = testing_data["Time"].values.reshape(-1, 1)
testing_output_data = testing_data["output"].values.reshape(-1, 1)

# function predicts the output of the solar panels based on the time
predict_output()

file_name = "datasetanoon.csv"
testing_data = pd.read_csv("testinganoon.csv")

testing_time_data = testing_data["Time"].values.reshape(-1, 1)
testing_output_data = testing_data["output"].values.reshape(-1, 1)

predict_output()
