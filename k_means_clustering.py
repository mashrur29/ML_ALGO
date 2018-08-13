import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

import sklearn
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report

# Matplotlib inline
rcParams['figure.figsize'] = 7, 4

iris = datasets.load_iris()

X = scale(iris.data)
y = pd.DataFrame(iris.target)
variable_names = iris.feature_names

# print(X[0:, ])

clustering = KMeans(n_clusters=3, random_state=5)
clustering.fit(X)

