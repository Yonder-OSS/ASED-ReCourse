import pandas as pd
import numpy as np
from Sloth import Sloth

import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


from tslearn.preprocessing import TimeSeriesScalerMeanVariance

Sloth = Sloth()
datapath = 'post_frequency_8.09_8.15.csv'
series = pd.read_csv(datapath,dtype='str',header=0)
series = series.rename(index=str, columns={0: 'usernames'})

#print("DEBUG::post frequency data:")
#print(series)

# scaling can sometimes improve performance
X_train = series.values[:,1:]

#X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)

n_samples = 100
X_train = X_train[:n_samples]
X_train = X_train.astype(np.float)

usernames = series.values[:n_samples,0]

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
    SimilarityMatrix = Sloth.LoadSimilarityMatrix()    
else:
    SimilarityMatrix = Sloth.GenerateSimilarityMatrix(X_train)
    Sloth.SaveSimilarityMatrix(SimilarityMatrix)

nclusters, labels, cnt = Sloth.ClusterSimilarityMatrix(SimilarityMatrix,eps,min_samples)

## this is the second clustering method, using tslearn kmeans
'''nclusters = 10
labels = Sloth.ClusterSeriesKMeans(X_train,nclusters)'''

## plot clustering results in the time dimension
series_np = X_train
plt.figure()
for yi in range(nclusters):
    plt.subplot(nclusters, 1, 1 + yi)
    for xx in series_np[labels == yi]:
        plt.plot(xx.ravel(), "k-")
    plt.xlim(0, ncols)
    plt.title("Cluster %d: %d series" %(yi,cnt[yi]))

clust = 3

print("DEBUG::anomalous series in selected number %d cluster:"%clust)
print(series_np[labels==clust])
print(usernames[labels==clust])

print("DEBUG::AniyaHadlee cluster:")
print(labels[usernames=='AniyaHadlee'])
print("DEBUG::BarackDonovan cluster:")
print(labels[usernames=='BarackDonovan'])
print("DEBUG::BlameItonBHO cluster:")
print(labels[usernames=='BlameItonBHO'])

print("DEBUG::Distance between AniyaHadlee and BarackDonovan:")
print(SimilarityMatrix[np.arange(n_samples)[usernames=='BarackDonovan'],np.arange(n_samples)[usernames=='AniyaHadlee']])


print("DEBUG::cluster %d series:"%clust)
print(series_np[labels==clust])
print(usernames[labels==clust])

plt.tight_layout()
plt.show()

## plot in frequency dimension? -- fourier transform, etc.

## generate a graph based on results, and plot it
import networkx as nx

print("DEBUG::distance matrix:")
print(SimilarityMatrix)

dist_threshold = eps

edge_list_names = []
edge_nodes_names = []
edge_nodes_labels = []
for j in np.arange(nrows):
    for i in np.arange(ncols):
        if(j>i):
            if(SimilarityMatrix[j,i]<=dist_threshold):
                edge_list_names.append((usernames[j],usernames[i]))
                if usernames[j] not in edge_nodes_names:
                    edge_nodes_names.append(usernames[j])
                    edge_nodes_labels.append(labels[j])
                if usernames[i] not in edge_nodes_names:
                    edge_nodes_names.append(usernames[i])
                    edge_nodes_labels.append(labels[i])

print("DEBUG::number of edges in graph:")
print(len(edge_list_names))
print("DEBUG::the nodes with edges are:")
print(edge_nodes_names)
print("DEBUG::there are %d such nodes"%len(edge_nodes_names))
print("DEBUG::the labels of nodes with edges are:")
print(edge_nodes_labels)
print("DEBUG::there are %d such labels"%len(edge_nodes_labels))

G_users=nx.Graph()
G_users.add_nodes_from(usernames)
#print("DEBUG::all nodes in graph:")
#print(G_users.nodes())
G_users.add_edges_from(edge_list_names)
plt.figure()
nx.draw_spring(G_users,with_labels=True)
plt.show()

nx.write_edgelist(G_users,"G_users_complete_NAMES.edgelist",data=False)

usernames_idx = np.arange(len(edge_nodes_names)) #nodes are renumbered as contiguous integers

edge_list = []
for j in np.arange(len(edge_nodes_names)):
    for i in np.arange(len(edge_nodes_names)):
        if(j>i):
            if(SimilarityMatrix[usernames==edge_nodes_names[j],usernames==edge_nodes_names[i]]<=dist_threshold):
                edge_list.append((usernames_idx[j],usernames_idx[i]))

G_users=nx.Graph()
G_users.add_nodes_from(usernames_idx)
G_users.add_edges_from(edge_list)
plt.figure()
nx.draw_spring(G_users,with_labels=True)
plt.show()

nx.write_edgelist(G_users,"G_users_trimmed_INDEXED.edgelist",data=False)


import os
os.environ["THEANO_FLAGS"]="mode=FAST_RUN,device=gpu,floatX=float32"

from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
from time import time

from gem.embedding.gf       import GraphFactorization
from gem.embedding.hope     import HOPE
from gem.embedding.lap      import LaplacianEigenmaps
from gem.embedding.lle      import LocallyLinearEmbedding

# File that contains the edges. Format: source target
# Optionally, you can add weights as third column: source target weight
edge_f = 'G_users_trimmed_INDEXED.edgelist'
# Specify whether the edges are directed
isDirected = True

# Load graph
G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
G = G.to_directed()

models = []
# You can comment out the methods you don't want to run
models.append(GraphFactorization(d=2, max_iter=100000, eta=1*10**-4, regu=1.0))
models.append(HOPE(d=4, beta=0.01))
#models.append(LaplacianEigenmaps(d=2))
#models.append(LocallyLinearEmbedding(d=2))

for embedding in models:
    print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
    t1 = time()
    # Learn embedding - accepts a networkx graph or file with edge list
    Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
    print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
    # Evaluate on graph reconstruction
    MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
    #---------------------------------------------------------------------------------
    print(("\tMAP: {} \t precision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
    #---------------------------------------------------------------------------------
    # Visualize
    viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=edge_nodes_labels)
    plt.show()
    viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
    plt.show()

