import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

# sklearn provides the housing dataset
from sklearn.datasets import load_boston

boston = load_boston()

# print(boston)

# Load the data file into Feature df_x = X, and target df_y = y

df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
df_y = pd.DataFrame(boston.target)

# print(df_x.describe())

# Split the data into test and train

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

# Deploy model

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train) # Train Model

print(reg.coef_) # Theta value

# Predict Using test set

a =  reg.predict(x_test)
print(a) # predicted output

# Mean square error

error = np.mean((a - y_test)**2)
print(error)