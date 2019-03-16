import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
import pandas as pd

dataset = pd.read_csv('resources\\cars_prices_multi.csv')
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 2].values
print(dataset, X, y)
model = sklearn.linear_model.LinearRegression()
model.fit(X,y)
x_new = [[9.3,1970]]
prediction = model.predict(x_new)
print(prediction)