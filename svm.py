import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100)

X = digits.data[:, :]
y = digits.target[:]
clf.fit(X, y)

res = clf.predict(digits.data)

print('Prediction: ', res[2])

plt.imshow(digits.images[2], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()