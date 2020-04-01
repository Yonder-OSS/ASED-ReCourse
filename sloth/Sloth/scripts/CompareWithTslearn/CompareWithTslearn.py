import numpy
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tslearn.clustering import GlobalAlignmentKernelKMeans
from tslearn.metrics import sigma_gak, cdist_gak
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import pandas as pd
import numpy as np
from Sloth import cluster

seed = 0
numpy.random.seed(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X_train = X_train[y_train < 4]  # Keep first 3 classes
numpy.random.shuffle(X_train)

#X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])  # Keep only 50 time series

X_train = X_train[:50]

sz = X_train.shape[1]

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]))

#Sloth = Sloth()
eps = 20
min_samples = 2
LOAD = False # Flag for loading similarity matrix from file if it has been computed before
if(LOAD):
    SimilarityMatrix = cluster.LoadSimilarityMatrix()    
else:
    SimilarityMatrix = cluster.GenerateSimilarityMatrix(X_train)
    cluster.SaveSimilarityMatrix(SimilarityMatrix)
nclusters, labels, cnt = cluster.ClusterSimilarityMatrix(SimilarityMatrix,eps,min_samples)

print("DEBUG::number of clusters found =")
print(nclusters)

plt.figure()
for yi in range(nclusters):
    plt.subplot(nclusters, 1, 1 + yi)
    for xx in X_train[labels == yi]:
        plt.plot(xx.ravel(), "k-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.title("Cluster %d" % (yi + 1))

plt.tight_layout()
plt.show()