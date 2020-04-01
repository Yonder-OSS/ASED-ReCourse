# Sloth
Sloth - Strategic Labeling Over Time Heuristics - Tools for time series analysis

This library contains five modules: visualize, preprocess, cluster, classify, and predict. **Sloth.preprocess** contains methods for [Gaussian smoothing] and [Fourier transformation]. **Sloth.cluster** contains tools contains methods for generating a similarity matrix, [DBSCAN clustering], [HDBSCAN clustering], and KMEANS clustering. **Sloth.classify** contains the **Shapelets** class, which learns representative [time series subsequences] for time series classification. (these are not unique solutions) and a method for KNN classification. **Sloth.predict** contains methods for [seasonal decomposition], and [ARIMA prediction]. 

## Available Functions

## Sloth.visualize

#### VisuallyCompareTwoSeries
Inputs = time series, index at which to begin visualization, index at which to end visualization

## Sloth.preprocess

#### events_to_rates
Inputs = list of event times, length of Gaussian filter bandwith, number of bins, maximum time. Outputs = list of rate values smoothed by a Gaussian filter and their corresponding rate times

#### rand_fourier_features(rate_vals, dim=1000, random_state=0):
Inputs = list of rate values, dimensionality of computed feature space, (optional) random state. Outputs = a feature map of an RBF kernel computed by a Monte Carlo approximation of its Fourier transform. 

## Sloth.cluster

#### GenerateSimilarityMatrix
Inputs = matrix of time series, where the n_rows = the number of series and n_cols = the length of each series. Output = similarity matrix between series after dynamic time warping transformation.

#### ClusterSimilarityMatrix
Inputs = similarity matrix, eps (maximum distance between two samples to be considered in same neighborhood), minimum number of samples. Outputs = number of clusters, label of cluster for each time series, count of time series in each cluster. 

#### HClusterSimilarityMatrix
Inputs = similarity matrix, minimum number of samples. Outputs = number of clusters, label of cluster for each time series, count of time series in each cluster. 

#### SaveSimilarityMatrix
Inputs = similarity matrix, filename. 

#### SaveSparseSimilarityMatrix
Inputs = similarity matrix, filename.

#### LoadSimilarityMatrix
Inputs = filename. Outputs = similarity matrix.

#### ClusterSeriesKMeans
Inputs = matrix of time series, number of clusters, algorithm (options are 'GlobalAlignmentKernelKMeans' and 'TimeSeriesKMeans'). Output = predicted cluster of each time series. 

## Sloth.classify

## Shapelets

#### PredictClasses
Inputs = time series. Output = classification of time series as one of time series in training data. 

#### VisualizeShapelets
Inputs = none. Output = graphs of learned shapelets

#### VisualizeShapeletLocations
Inputs = time series, id of time series from training data. Output = graph of learned shapelets superimposed on indexed time series from training data. 

## Sloth.predict

#### DecomposeSeriesSeasonal
Inputs = time series index, time series, (optionally) frequency of time series. Outputs = object with observed, trend, seasonal, and residual components. 

#### FitSeriesArima
Inputs = data frame containing two columns, 1) time series index and 2) time series values, whether the time series is seasonal, and (optionally) the period of seasonal differencing. Outputs = ARIMA model fit on the input data

#### PredictSeriesArima
Inputs = number of periods to predict in the future. Outputs = time series prediction for the number of periods to predict.

[Gaussian smoothing]: https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.ndimage.filters.gaussian_filter1d.html
[Fourier transformation]: https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html
[DBSCAN clustering]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
[HDBSCAN clustering]: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
[seasonal decomposition]: https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html
[ARIMA prediction]: https://www.alkaline-ml.com/pyramid/modules/generated/pyramid.arima.auto_arima.html
[time series subsequences]: http://fs.ismll.de/publicspace/LearningShapelets/

