import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import sparse
import hdbscan
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from collections import Counter
from tslearn.clustering import TimeSeriesKMeans, GlobalAlignmentKernelKMeans
from tslearn.metrics import sigma_gak, cdist_gak

def GenerateSimilarityMatrix(series):
    nrows,_ = series.shape
    # now, compute the whole matrix of similarities 
    print("Computing similarity matrix...")
    try:
            distances = [[fastdtw(series[j,:], series[i,:],dist=euclidean)[0] for i in range(j, nrows)] for j in np.arange(nrows)]
    except Exception as e:
        print(e)
        pass

    SimilarityMatrix = np.array([[0]*(nrows-len(i)) + i for i in distances])
    SimilarityMatrix[np.tril_indices(nrows,-1)] = SimilarityMatrix.T[np.tril_indices(nrows,-1)]
    print("DONE!")

    return SimilarityMatrix

def ClusterSimilarityMatrix(SimilarityMatrix,eps,min_samples):
    # perform DBSCAN clustering
    db = DBSCAN(eps=eps,min_samples=min_samples,metric='precomputed')
    db.fit(SimilarityMatrix)
    labels = db.labels_
    nclusters = len(set(labels))-(1 if -1 in labels else 0)
    cnt = Counter()
    for label in list(labels):
        cnt[label] += 1

    return nclusters, labels, cnt

def HClusterSimilarityMatrix(SimilarityMatrix,min_cluster_size, min_samples,PLOT=False):
    # perform DBSCAN clustering
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,metric='precomputed')
    labels = hdb.fit_predict(SimilarityMatrix)
    nclusters = len(set(labels))-(1 if -1 in labels else 0)
    cnt = Counter()
    for label in list(labels):
        cnt[label] += 1
    if(PLOT):
        plt.figure()
        hdb.condensed_tree_.plot()
        plt.figure()
        hdb.single_linkage_tree_.plot(cmap='viridis',colorbar=True)

    return nclusters, labels, cnt

def SaveSimilarityMatrix(SimilarityMatrix,filename):
    np.save(filename,SimilarityMatrix)

def SaveSparseSimilarityMatrix(SimilarityMatrix,filename):
    # sometimes the following may make sense - create a sparse representation
    SimilarityMatrixSparse = sparse.csr_matrix(SimilarityMatrix)
    with open(filename,'wb') as outfile:
        pickle.dump(SimilarityMatrixSparse,outfile,pickle.HIGHEST_PROTOCOL)

def LoadSimilarityMatrix(filename):
    SimilarityMatrix = np.load(filename+'.npy')
    return SimilarityMatrix

class KMeans():
    def __init__(self, n_clusters, algorithm='GlobalAlignmentKernelKMeans', random_seed = 0):
        '''
            initialize KMeans clustering model with specific kernel

            hyperparameters:
                n_clusters:         number of clusters in Kmeans model
                algorithm:          which kernel to use for model, options 
                                    are 'GlobalAlignmentKernelKMeans' and 'TimeSeriesKMeans'
                random_seed:        random seed with which to initialize Kmeans
        '''
        try:
            assert algorithm == 'GlobalAlignmentKernelKMeans' or algorithm == 'TimeSeriesKMeans'
        except:
            raise ValueError("algorithm must be one of \'GlobalAlignmentKernelKMeans\' or \'TimeSeriesKMeans\'")
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.algorithm = algorithm
        self.km = None

    def fit(self, train):
        '''
            fit KMeans clustering model on training data

            parameters:
                train                : training time series
        ''' 

        if self.algorithm == 'TimeSeriesKMeans':
            self.km = TimeSeriesKMeans(n_clusters=self.n_clusters, n_init=20, verbose=True, random_state=self.random_seed)
        else:
            self.km = GlobalAlignmentKernelKMeans(n_clusters=self.n_clusters, sigma=sigma_gak(train), n_init=20, verbose=True, random_state=self.random_seed)
        self.km.fit(train)

    def predict(self, test):
        '''
            clusters for time series in test data set

            parameters:
                test:     test time series on which to predict clusters

            returns: clusters for test data set
        '''
        return self.km.predict(test)
