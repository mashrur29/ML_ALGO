import numpy as np
import pandas as pd

from pandas import Series, DataFrame

import scipy
from scipy.stats import spearmanr

from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report

# matplotlib inline
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

# Input data from csv file

address = 'E:\Projects\ML_ALGO\Dataset\mtcars.csv'
cars = pd.read_csv(address)
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']

# Feature Selection X = [col 5, col 11], Y = [col 9]
# 0-Based Index

cars_data = cars.ix[:, (5, 11)].values
cars_data_names = ['drat', 'carb']
y = cars.ix[:, 9].values

# Scatter plot to check for ordinal value (checking Independence between features)

sb.regplot(x='drat', y='carb', data=cars, scatter=True)
# plt.show() # Uncomment to display

# Checking Independence between features, if spearman coefficient is low then less co-related

drat = cars['drat']

carb = cars['carb']

spearmanr_coefficient, p_value = spearmanr(drat, carb)
print('Spearman Rank Corelation Coefficient: %0.3f' % spearmanr_coefficient)


# Checking for missing values

print(cars.isnull().sum(0))  # sum of null values, ) if none

# Check whether target is binary/ordinal using Histogram

sb.countplot(x='am', data=cars, palette='hls')
# plt.show() # Uncomment to display


# Checking Whether the size of data-set is sufficient
# Atleast 50 observations

print(cars.info())

# Deployment and Evaluation

X = scale(cars_data)
LogReg = LogisticRegression()

LogReg.fit(X, y)

print(LogReg.score(X, y)) # 0 - 1, 1 for a perfect fit

# Here X_test is the data which we will predict

# Input test file

address_test = 'E:\Projects\ML_ALGO\Dataset\mtcarstest.csv'
cars_test_file = pd.read_csv(address_test)
cars_data_test = cars_test_file.ix[:, (0, 1)].values
X_test = scale(cars_data_test)

# Predict and check score

y_pred = LogReg.predict(X)  # Here I used X as i wanted to know the error from report by comparison
                            # Use X_test for a test file

print(classification_report(y, y_pred)) # length(y_pred) = length(y)







