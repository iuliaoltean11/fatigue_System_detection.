import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#load dataset
dataset = pd.read_csv('SVM_training_data_2.csv')

#split dataset in x si y
training_vector = dataset.iloc[:, [0, 1]].values
target_values = dataset.iloc[:, 2].values
print(target_values)

#perform feature scaling
sc = StandardScaler()
training_vector = sc.fit_transform(training_vector)

#fit svm to data
classifier = SVC(kernel = 'rbf', random_state = 0)
clf = classifier.fit(training_vector, target_values)

#predict class for new data
new_data = [0.8, 0.9]
new_data = sc.transform([new_data])
predicted_class = classifier.predict(new_data)
print(predicted_class)

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# model = svm.SVC(kernel='linear')
# clf = model.fit(X, y)

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = training_vector[:, 0], training_vector[:, 1]
xx, yy = make_meshgrid(X0, X1)

colors = ['white', 'grey']
cmap = ListedColormap(colors)

plot_contours(ax, clf, xx, yy, cmap=cmap, alpha=0.8)
ax.scatter(X0, X1, c=target_values, cmap='RdYlBu', s=20, edgecolors='k')
ax.set_ylabel('Cadrul 2')
ax.set_xlabel('Cadrul 1')
# ax.set_xticks((1, 2, 3))
# ax.set_yticks((1, 2,3 ))
ax.set_title(title)
# ax.legend(("a", "b"))
# ax.grid(True)
plt.show()