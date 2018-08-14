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
# print(clustering.labels_)     # Print Labels

# Plotting Model Output

iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y.columns = ['Target']


relabel = np.choose(clustering.labels_, [2, 0, 1]).astype(np.int64)     # Relabel to match with output color, this is optional

# relabel is the output of clustering

color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])

plt.subplot(1, 2, 1)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title('Ground Truth Classification')

plt.subplot(1, 2, 2)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[relabel], s=50)
plt.title('K-Means Clustering')

plt.show()

print(classification_report(y, relabel))

