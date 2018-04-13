from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('/home/soumen/Desktop/TML_HW3_AT/winequalityRed.csv', header = None)
print("Shape of the training data is {}".format(data.shape))
[m ,n] = data.shape
X = data.iloc[:, 0:n-1].values
y = data.iloc[:, n-1:n].values

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1)

test_list = [0]
train_list = [0]

for i in range(1, 3):
    poly = PolynomialFeatures(degree=i)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)

    clf = linear_model.LinearRegression()
    clf.fit(x_train_poly, y_train)
    print("The degree of the polynomial is {}".format(i))
    test_yhat = clf.predict(x_test_poly)
    train_yhat = clf.predict(x_train_poly)
    test_mse = mean_squared_error(y_test, test_yhat)
    train_mse = mean_squared_error(y_train, train_yhat)
    test_list.append(test_mse)
    train_list.append(train_mse)
    print("The test Accuracy is {}".format(test_mse))
    print("The training Accuracy is {}".format(train_mse))
    print('.............................................')


plt.figure(1)
plt.subplot(211)
plt.plot(train_list)
plt.ylabel('Training Error(MSE)')
plt.xlabel('Degree of the polynomial')
plt.grid(True)

plt.subplot(212)
plt.plot(test_list)
plt.ylabel('Testing Error(MSE)')
plt.xlabel('Degree of the polynomial')
plt.grid(True)
plt.show()
