import numpy as np
import pandas as pd
import pickle
from keras.optimizers import Adam, Adagrad, RMSprop
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

class Shapelets():
    def __init__(self, epochs, length, num_shapelet_lengths, num_shapelets, learning_rate, weight_regularizer, 
        batch_size = 256, optimizer = Adam):
        '''
            initialize shapelet hyperparameters

            hyperparameters:
                epochs                : number of training epochs
                length                : base shapelet length, expressed as fraction of length of time series
                num_shapelet_lengths  : number of different shapelet lengths
                num_shapelets         : number of unique shapelets to learn at each shapelet length, 
                                        expressed as fraction of length of time series
                learning rate         : learning rate of Keras optimizer
                weight regularizer    : weight regularization used when fitting model
        '''
        self.epochs = epochs
        self.length = length
        self.num_shapelet_lengths = num_shapelet_lengths
        self.num_shapelets = num_shapelets
        self.weight_regularizer = weight_regularizer
        self.batch_size = batch_size
        self.optimizer = optimizer(lr = learning_rate)
        self.shapelet_sizes = None
        self.shapelet_clf = None
        self.encoder = LabelEncoder()

    def clear_session(self):
        try:
            assert(self.shapelet_clf is not None)
        except:
            raise ValueError("Cannot clear session that has not been initialized")
        self.shapelet_clf.clear_session()
        return

    def load_model(self, series_length, labels, checkpoint):
        '''
            Load model from checkpoint into Shapelet classifier
        '''
        if self.shapelet_clf is None:
            base_size = int(self.length * series_length)
            self.shapelet_sizes = {}
            for sz_idx in range(self.num_shapelet_lengths):
                shp_sz = base_size * (sz_idx + 1)
                self.shapelet_sizes[shp_sz] = int(self.num_shapelets * series_length)
            self.shapelet_clf = ShapeletModel(n_shapelets_per_size=self.shapelet_sizes,
                            optimizer=self.optimizer,
                            weight_regularizer=self.weight_regularizer,
                            max_iter=self.epochs, batch_size=self.batch_size)
        
        # first generate new model into which to load the weights
        self.encode(labels)
        self.shapelet_clf.generate_model(series_length, len(self.get_classes()))

        # load weights
        self.shapelet_clf.model.load_weights(checkpoint)

    def fit_transfer_model(self, X_train, y_train, checkpoint, nclasses_prior = 2, source_dir = None, val_data = None):        
        # encode training and validation labels
        y_train = self.encode(y_train)
        y_val = self.encode(val_data[1])

        # scale training and validation data to between 0 and 1
        X_train_scaled = self.__ScaleData(X_train)
        X_val_scaled = self.__ScaleData(val_data[0])

        if self.shapelet_clf is None:
            base_size = int(self.length * X_train.shape[1])
            self.shapelet_sizes = {}
            for sz_idx in range(self.num_shapelet_lengths):
                shp_sz = base_size * (sz_idx + 1)
                self.shapelet_sizes[shp_sz] = int(self.num_shapelets * X_train.shape[1])
            self.shapelet_clf = ShapeletModel(n_shapelets_per_size=self.shapelet_sizes,
                                optimizer=self.optimizer,
                                weight_regularizer=self.weight_regularizer,
                                max_iter=self.epochs, batch_size=self.batch_size)

        # fit shapelet classifier
        self.shapelet_clf.fit_transfer_model(X_train_scaled, y_train, nclasses_prior, checkpoint, source_dir, (X_val_scaled, y_val))

    def fit(self, X_train, y_train, source_dir = None, val_data = None):
        '''
            fit shapelet classifier on training data

            parameters:
                X_train                : training time series
                y_train                : training labels
        ''' 
        if self.shapelet_clf is None:
            base_size = int(self.length * X_train.shape[1])
            self.shapelet_sizes = {}
            for sz_idx in range(self.num_shapelet_lengths):
                shp_sz = base_size * (sz_idx + 1)
                self.shapelet_sizes[shp_sz] = int(self.num_shapelets * X_train.shape[1])
            self.shapelet_clf = ShapeletModel(n_shapelets_per_size=self.shapelet_sizes,
                                optimizer=self.optimizer,
                                weight_regularizer=self.weight_regularizer,
                                max_iter=self.epochs, batch_size=self.batch_size)
        
        # encode training and validation labels
        y_train = self.encode(y_train)
        y_val = self.encode(val_data[1])

        # scale training and validation data to between 0 and 1
        X_train_scaled = self.__ScaleData(X_train)
        X_val_scaled = self.__ScaleData(val_data[0])

        # fit classifier
        self.shapelet_clf.fit(X_train_scaled, y_train, source_dir, (X_val_scaled, y_val))
        
    def __ScaleData(self, input_data):
        ''' 
            scale input data to range [0,1]

            parameters:
                input_data        : input data to rescale
        '''

        return TimeSeriesScalerMinMax().fit_transform(input_data)

    def predict(self, X_test):
        '''
            classifications for time series in test data set

            parameters:
                X_test:     test time series on which to predict classes

            returns: classifications for test data set
        '''
        X_test_scaled = self.__ScaleData(X_test)
        return self.shapelet_clf.predict(X_test_scaled) 

    def predict_proba(self, X_test):
        '''
            class probabilities for time series in test data set

            parameters:
                X_test:     test time series on which to predict classes

            returns: classifications for test data set
        '''
        X_test_scaled = self.__ScaleData(X_test)
        return self.shapelet_clf.predict_proba(X_test_scaled) 

    def encode(self, categories):
        '''
            fit label encoder on input categories. returns transformed categories
        '''
        self.encoder.fit(categories)
        return self.encoder.transform(categories)

    def decode(self, y_probs, p_threshold):
        '''
            decode prediction probabilities y_probs into prediction / confidence give p_threshold
        '''
        prob_max = np.amax(y_probs, axis = 1)
        prediction_indices = prob_max > p_threshold
        y_pred = np.zeros(y_probs.shape[0])
        
        # reintepret confidence in binary case
        if y_probs.shape[1] == 1:
            y_pred[prediction_indices] = 1 
            confidence = (prob_max - p_threshold) / (y_pred - p_threshold)
            confidence = 0.5 + confidence / 2
        else:
            y_pred[prediction_indices] = np.argmax(y_probs, axis = 1)[prediction_indices]
            confidence = prob_max
        y_pred = y_pred.astype(int)
        y_preds = self.encoder.inverse_transform(y_pred)

        return y_preds, confidence
    
    def get_classes(self):
        '''
            get original classes from encoder
        '''
        try:
            assert(self.encoder is not None)
        except:
            raise ValueError("Encoder has not been initialized")
        return self.encoder.classes_

    def VisualizeShapelets(self):
        '''
            visualize all of shapelets learned by shapelet classifier
        '''
        plt.figure()
        for i, sz in enumerate(self.shapelet_sizes.keys()):
            plt.subplot(len(self.shapelet_sizes), 1, i + 1)
            plt.title("%d shapelets of size %d" % (self.shapelet_sizes[sz], sz))
            for shapelet in self.shapelet_clf.shapelets_:
                if ts_size(shapelet) == sz:
                    plt.plot(shapelet.ravel())
            plt.xlim([0, max(self.shapelet_sizes.keys())])
        plt.tight_layout()
        plt.show() 

    def VisualizeShapeletLocations(self, series_values, series_id, save_dir = 'visualizations', 
        name = 'shp_1'):
        '''
            visualize shapelets superimposed on one of the test series

            parameters:
                series_values:      raw values on which to visualize shapelets
                series_id:          id of time series to visualize  
                save_dir            directory in which to save visualizations
                name                name under which to save viz (bc unique every time)
                n_shapelets:        
        '''

        plt.style.use("seaborn-whitegrid")

        # NK brand colors
        COLORS = ["#FA5655", "#F79690", "#B9BC2D", "#86B6B2", "#955B99", "#252B7A"]
        # others? "#8D6B2C",
                # "#D0A826",
                # "#FEDB03",
                # "#000000",
                # "#454545",
                # "#FFFFFF",
                # "#F8F6F1"]
        n_rows, n_cols, _ = series_values.shape
        test_series = series_values[series_id].reshape(-1,)

        closest_inds = self.shapelet_clf.locate(test_series.reshape(1,-1,1))[0]
        closest_dists = []
        for ind , shp in zip(closest_inds, self.shapelet_clf.shapelets_):
            closest_dists.append(np.linalg.norm(test_series[ind : ind + shp.shape[0]] - shp))
        closest_dists = np.array(closest_dists)

        # convert distance to weight where dist=0 -> wt=1 and dist=inf -> wt=0
        sl_weights = 1 / (1 + closest_dists)
        # plot the signal with matching shapelets color overlayed
        plt.clf()
        plt.plot(range(n_cols), test_series, color="k")
        for ind, sl, wt, color in zip(closest_inds, self.shapelet_clf.shapelets_, sl_weights, COLORS):
            # find closest match
            t = range(ind, ind + sl.shape[0])
            match = test_series[ind : ind + sl.shape[0]]
            # plot shapelet on top of signal width width and alpha set by shapelet weight
            plt.plot(t, match, alpha=7 * wt, linewidth=35 * wt, color=color)
        plt.ylabel('Email Density')
        plt.xlabel('Minute of the Hour')
        plt.show()
        #plt.savefig(save_dir + "/{}_signal_size_{}_id_{}.png".format(name, n_cols, series_id))

        # plot shapelets
        plt.clf()
        # to plot the shapelets, switch to dark background
        plt.style.use("seaborn-darkgrid")
        # ax = plt.axes()  # used below for sharex, sharey (if needed?)

        # arange shapletes in grid - find greatest factor of n_shapelets
        gf = 0
        shp_t = self.shapelet_clf.shapelets_as_time_series_
        shp = self.shapelet_clf.shapelets_
        for i in range(1,shp.shape[0]):
            if shp.shape[0]%i==0:
                gf = i
        of = int(shp.shape[0] / gf)
        n_cols = 2
        for i in range(shp_t.shape[0]):
            ax_i = plt.subplot(gf, of, i + 1)
            # we could force them to share the same axes
            # ax_i = plt.subplot(n_rows, n_cols, i + 1, sharex=ax, sharey=ax)
            #ax_i.set_xticklabels([])
            ax_i.set_yticklabels([])
            plt.plot(range(shp_t.shape[1]), shp[i].reshape(-1), color=COLORS[i%len(COLORS)], linewidth=3)
            plt.xlabel('Shapelet Length')
        plt.show()
        #plt.savefig(save_dir + "/{}_shapelets_size_{}_id_{}.png".format(name, n_cols, series_id))

class Knn():
    def __init__(self, n_neighbors):
        '''
            initialize KNN class with dynamic time warping distance metric

            hyperparameters:
                n_neighbors           : number of neighbors on which to make classification decision
        '''
        self.n_neighbors = n_neighbors
        self.knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=n_neighbors, metric="dtw")

    def __ScaleData(self, input_data):
        ''' 
            scale input data to range [0,1]

            parameters:
                input_data        : input data to rescale
        '''

        return TimeSeriesScalerMinMax().fit_transform(input_data)

    def fit(self, X_train, y_train):
        '''
            fit KNN classifier on training data

            parameters:
                X_train                : training time series
                y_train                : training labels
        ''' 
        # scale training data to between 0 and 1
        X_train_scaled = self.__ScaleData(X_train)
        self.knn_clf.fit(X_train_scaled, y_train)
    
    def predict(self, X_test):
        '''
            classifications for time series in test data set

            parameters:
                X_test:     test time series on which to predict classes

            returns: classifications for test data set
        '''
        # scale test data to between 0 and 1
        X_test_scaled = self.__ScaleData(X_test)
        return self.knn_clf.predict(X_test_scaled) 

#   test using Trace dataset (Bagnall, Lines, Vickers, Keogh, The UEA & UCR Time Series
#   Classification Repository, www.timeseriesclassification.com
if __name__ == '__main__':

    # constants
    epochs = 200
    shapelet_length = 0.1
    num_shapelet_lengths = 2
    time_series_id = 0
    learning_rate = .01
    weight_regularizer = .01

    # create shapelets
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    trace_shapelets = Shapelets(epochs, shapelet_length, num_shapelet_lengths, learning_rate, weight_regularizer)
    trace_shapelets.fit(X_train, y_train)

    # test methods
    predictions = trace_shapelets.predict(X_test)
    print("Shapelet Accuracy = ", accuracy_score(y_test, predictions))
    trace_shapelets.VisualizeShapelets()
    trace_shapelets.VisualizeShapeletLocations(X_test, time_series_id)

    # test KNN classifier
    knn_clf = Knn(n_neighbors = 3)
    knn_clf.fit(X_train, y_train)
    knn_preds = knn_clf.predict(X_test)
    print("KNN Accuracy = ", accuracy_score(y_test, knn_preds))

