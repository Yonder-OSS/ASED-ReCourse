import pandas as pd
import numpy as np
from Sloth import cluster

import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# some hyper-parameters
eps = 20
min_samples = 2

datapath = '/data/home/jgleason/Downloads/UCR_TS_Archive_2015/synthetic_control/synthetic_control_TRAIN'
series = pd.read_csv(datapath,dtype='float',header=None)
nrows,ncols = series.shape
print("--DEBUG--")
print("nrows = %d" % nrows)
print("ncols = %d" % ncols)

# scaling can sometimes improve performance
X_train = series.values
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]))
X_train = X_train[:,1:]


LOAD = False # Flag for loading similarity matrix from file if it has been computed before
if(LOAD):
    SimilarityMatrix = cluster.LoadSimilarityMatrix()    
else:
    SimilarityMatrix = cluster.GenerateSimilarityMatrix(X_train[:200])
    cluster.SaveSimilarityMatrix(SimilarityMatrix, 'sim_matrix_new')

nclusters, labels, cnt = cluster.ClusterSimilarityMatrix(SimilarityMatrix,eps,min_samples)

series_np = series[:100].values

plt.figure()
for yi in range(nclusters):
    plt.subplot(nclusters, 1, 1 + yi)
    for xx in series_np[labels == yi]:
        plt.plot(xx.ravel(), "k-")
    plt.xlim(0, ncols)
    plt.ylim(-4, 4)
    plt.title("Cluster %d" % (yi + 1))

plt.tight_layout()
plt.show()

