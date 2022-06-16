from collections.abc import Iterable
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
from sklearn.kernel_approximation import RBFSampler

def events_to_rates(event_times, filter_bandwidth=1, num_bins=72, min_time = None,max_time=None, density = True):
    """ convert list of event times into rate function with a discrete time bin_size of 1/rates_per_unit.
    Uses a guassian filter over the empirical rate (histogram count / bin_size) """

    if len(event_times) == 0:  # if event times is an empty list or array
        print("empty event_times list/array")
        return np.zeros(num_bins), np.zeros(num_bins)

    if not max_time:
        max_time = max(event_times)
    if not min_time:
        min_time = min(event_times)
    bins = np.linspace(min_time, max_time, num=num_bins + 1)
    rate_times = (bins[1:] + bins[:-1]) / 2

    bin_size = (max_time - min_time) / num_bins
    if density:
        counts = np.array(np.histogram(event_times, bins=bins)[0])
        sampled_rates = counts / sum(counts)
    else:
        counts = np.array(np.histogram(event_times, bins=bins)[0])
        sampled_rates = counts / bin_size
    rate_vals = gaussian_filter1d(sampled_rates, filter_bandwidth, mode="nearest")
    return rate_vals, rate_times

def rand_fourier_features(rate_vals, dim=1000, random_state=0):
    if rate_vals.ndim == 1:
        rate_vals = rate_vals[None, :]
    rand_fourier = RBFSampler(n_components=dim, random_state=random_state)
    return rand_fourier.fit_transform(rate_vals)
