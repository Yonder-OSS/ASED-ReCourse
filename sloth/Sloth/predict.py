from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima
import pandas as pd

def DecomposeSeriesSeasonal(series_time_index,series, *frequency):
    data = pd.DataFrame(series,index = series_time_index,columns=["Series"])

    # use additive model if negative values in time series
    model = 'multiplicative'
    if (min(series) <= 0):
        model = 'additive'
    
    # call seasonal_decompose with optional frequency parameter
    if not frequency:
        if isinstance(series_time_index, pd.DatetimeIndex):
            return seasonal_decompose(data, model=model)
        else:
            return seasonal_decompose(data, model=model, freq=1)
    else:
        return seasonal_decompose(data, model=model, freq=frequency[0])

class Arima():
    def __init__(self, seasonal, *seasonal_differencing):
        '''
            initialize ARIMA class

            hyperparameters:
                seasonal:                boolean indicating whether time series has seasonal component
                seasonal_differencing:   optional HP indicating the length of seasonal differencing
        '''
        self.seasonal = seasonal
        self.seasonal_differencing = seasonal_differencing
        self.arima_model = None

    def fit(self, train):
        '''
            fit ARIMA model on training data

            parameters:
                train                : training time series
        ''' 
        # default: annual data
        if not self.seasonal_differencing:
            self.arima_model = auto_arima(train, start_p=1, start_q=1,
                            max_p=5, max_q=5, m=1,
                            seasonal=self.seasonal,
                            d=None, D=1, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)
        # specified seasonal differencing parameter
        else:
            self.arima_model = auto_arima(train, start_p=1, start_q=1,
                            max_p=5, max_q=5, m=self.seasonal_differencing[0],
                            seasonal=self.seasonal,
                            d=None, D=1, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)
        self.arima_model.fit(train)

    def predict(self, n_periods):
        '''
            forecasts the time series n_periods into the future

            parameters:
                n_periods:     number of periods to forecast into the future

            returns: time series forecast n_periods into the future
        '''
        return self.arima_model.predict(n_periods = n_periods)
