from sklearn.cluster import KMeans
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
X = np.array([[1., 2.,3.], [1., 4.,3.], [1., 1.,3.],
              [6., 2.,3.], [2., 3.,3.], [4., 2.,3.],
              [8., 1.,3.], [4., 5.,3.], [5., 3.,3.],
              [3., 0.,3.], [5., 0.,3.], [6., 4.,3.],
              [4., 9.,3.], [6., 4.,3.], [3., 6.,3.],
              [5., 4.,3.], [9., 7.,3.], [2., 4.,3.],
              [6., 2.,3.], [4., 5.,3.], [8., 3.,3.]])

print(X.shape, X.dtype)

rice_cluster = KMeans(n_clusters=3)

rice_cluster.fit(X)

label = rice_cluster.labels_
x0 = X[label == 0]
x1 = X[label == 1]
x2 = X[label == 2]

print('label = ', label)

print("x0 = ", x0)
print("x1 = ", x1)
print("x2 = ", x2)