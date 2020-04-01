import numpy as np
#import matplotlib
#matplotlib.use('Agg') # uncomment for docker images
import matplotlib.pyplot as plt

def VisuallyCompareTwoSeries(series,i1,i2):
    fig = plt.figure()
    plt.title("Comparing Time Series")
    ax1 = fig.add_subplot(211)
    ax1.plot(np.arange(series.shape[1]-1), series.values[i1,1:])
    ax1.set_xlabel("time")
    ax1.set_ylabel(str(i1))
    ax2 = fig.add_subplot(212)
    ax2.plot(np.arange(series.shape[1]-1), series.values[i2,1:])
    ax2.set_xlabel("time")
    ax2.set_ylabel(str(i2))
    plt.show()