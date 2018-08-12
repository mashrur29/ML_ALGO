import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Read data file
address = 'E:\\Projects\\ML_ALGO\\Dataset\\train.csv'  # train.csv is handwritten digit data
datas = pd.read_csv(address)

# Feature and Target specification
df_x = datas.iloc[:, 1:]  # all rows from column 1->last
df_y = datas.iloc[:, 0]  # all rows, and only 0th column

# Split into train and test data

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)


# deploy model
nn = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(10, 15), random_state=1)  # sgd is a form of gradient descent
nn.fit(x_train, y_train)

# Test on test data to check for % error

prediction = nn.predict(x_test)
a = y_test.values

# compute correct prediction / total output

count = 0
for i in range(len(prediction)):
    if prediction[i] == a[i]:
        count = count + 1


print(count / len(prediction))  # percentage error



