import pandas as pd
import numpy as np
from Sloth import cluster
import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from collections import Counter


#Sloth = Sloth()
datapath = 'post_frequency_8.09_8.15.csv'
series = pd.read_csv(datapath,dtype=str,header=0)

#print("DEBUG::post frequency data:")
#print(series)

X_train = series.values[:,1:]
# scaling can sometimes improve performance
#X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)

n_samples = 100
X_train = X_train[:n_samples]
X_train = X_train.astype(np.float)

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]))

nrows,ncols = X_train.shape

print("DEBUG::shape of final data for clustering:")
print(X_train.shape)

## this is the first clustering method, via dbscan
# some hyper-parameters
eps = 90
min_samples = 2
LOAD = True # Flag for loading similarity matrix from file if it has been computed before
if(LOAD):
    SimilarityMatrix = cluster.LoadSimilarityMatrix('SimilarityMatrix')    
else:
    SimilarityMatrix = cluster.GenerateSimilarityMatrix(X_train[:,1:])
    cluster.SaveSimilarityMatrix(SimilarityMatrix,'SimilarityMatrix')

nclusters, labels, cnt = cluster.ClusterSimilarityMatrix(SimilarityMatrix,eps,min_samples)

print("The cluster frequencies are:")
print(cnt)
        
## try hierarchical clustering
# nclusters, labels, cnt = cluster.HClusterSimilarityMatrix(SimilarityMatrix,min_samples,PLOT=False)

# print("The hcluster frequencies are:")
# print(cnt)

## this is another clustering method, using tslearn kmeans
# nclusters = 10
# labels = cluster.ClusterSeriesKMeans(X_train,nclusters)
# nclusters = len(set(labels))-(1 if -1 in labels else 0)
# from collections import Counter
# cnt = Counter()
# for label in list(labels):
#     cnt[label] += 1

# print("The k-means frequencies are:")
# print(cnt)

# plot the latest clustering result
series_np = X_train
cnt_nontrivial = {x:cnt[x] for x in cnt if cnt[x]>1 and x!=-1}
plt.figure()
idx = 0
for yi in cnt_nontrivial.keys():
    plt.subplot(len(cnt_nontrivial), 1, 1 + idx)
    for xx in series_np[labels == yi]:
        plt.plot(xx.ravel(), "k-")
    plt.xlim(0, ncols)
    plt.title("Cluster %d: %d series" %(yi,cnt[yi]))
    idx = idx+1

clust = 9

print("DEBUG::AniyaHadlee cluster:")
print(labels[series.values[:n_samples,0]=='AniyaHadlee'])
print("DEBUG::BarackDonovan cluster:")
print(labels[series.values[:n_samples,0]=='BarackDonovan'])

print("DEBUG::cluster=%d series are:"%clust)
print(series_np[labels==clust]) # series
print(series.values[:n_samples,0][labels==clust]) # usernames

plt.tight_layout()
plt.show()