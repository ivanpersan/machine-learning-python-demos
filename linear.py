import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
import pandas as pd

dataset = pd.read_csv('resources\\cars_prices.csv')
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values
model = sklearn.linear_model.LinearRegression()
model.fit(X,y)
x_new = [[9]]
prediction = model.predict(x_new)
print(prediction)