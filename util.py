# System libraries
import os
import math

# Accelerate Scikit-learn with Intel extension: https://intel.github.io/scikit-learn-intelex
from sklearnex import patch_sklearn
patch_sklearn()

# Mertrics
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import mean_squared_error

# Imputation
from sklearn.impute import KNNImputer

# Peaks
from scipy.signal import find_peaks, find_peaks_cwt, peak_prominences

# Core libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm

# FancyImpute
from fancyimpute import IterativeImputer

# Plotting
import matplotlib.pyplot as plt
from pylab import rcParams
plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 18, 8

# IO
## Get server filenames for a datacenter
def get_servers(root, dc):
    filenames = []
    for path, _, files in os.walk('{}/{}'.format(root, dc)):
        for name in files:
            filenames.append(os.path.join(path, name))
    return filenames

## Load a time series
def load_file(filename, agg_func=np.mean, freq='H'):
    data = pd.read_csv(filename)
    data.rename({'Unnamed: 0': 'timestamp'}, axis=1, inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Index the data
    data.set_index('timestamp', inplace=True)
    data.index = data.index.round(freq)
    data = data.resample(freq).agg(agg_func)
    data.index.freq = freq
    return data

## Load multiple time series
def load_files(filenames, agg_func=np.mean, freq='H'):
    df = pd.DataFrame()
    for name in filenames:
        # Load data
        data = load_file(name, agg_func, freq)

        # Aggregate the values in one dataframe
        if df.empty:
            df = data.copy()
        else:
            if len(df) < len(data):
                df = df.reindex(index=data.index, method='nearest')
            elif len(df) > len(data):
                data = data.reindex(index=df.index, method='nearest')
            df = df.add(data)
    df[df < 0] = np.nan
    df = df.round(2)
    return df

## Load time series data for a dataceter
def load_data(root, dc, agg_func=np.mean, freq='H'):
    # Get server filenames
    filenames = get_servers(root, dc)

    return load_files(filenames, agg_func, freq)

## Cluster servers based on their seasonality
def group_servers(filenames, series, horizon, freq, agg_func=np.mean, load=True):
    groups = {}
    for name in filenames:
        # Load file
        df = load_file(name, agg_func, freq)

        # Preprocess
        ts_power = df[series]
        ts_power.index.freq = freq
        ts_power_train = ts_power[:ts_power.index[-1] - horizon]

        # Get periods
        periods = get_periods(ts_power_train, all=False)

        # Put in the right group
        if not periods:
            if 0 not in groups:
                groups[0] = [name]
            else:
                groups[0].append(name)
        else:
            period = periods[0]
            if period not in groups:
                groups[period] = [name]
            else:
                groups[period].append(name)

    # Return a dict of data frames
    if load:
        data = {}
        # Put in the right group
        for g in groups:
            filenames = groups[g]
            df = load_files(filenames, agg_func, freq)
            ts_power = df[series]
            ts_power.index.freq = freq
            if g in data:
                ts_power_old = data[g]
                data[g] = ts_power_old.add(ts_power)
            else:
                data[g] = ts_power
        return data
    return groups

# Peak detection
## Detect peaks
def detect_peaks(ts, lag, threshold, influence, k, h):
    peaks = {}
    peaks_zscore = zscore(ts, lag, threshold, influence)['signals']
    peaks_zcrossing = zero_crossing(ts,k, h, measure ='S1')
    peaks_scipy = find_peaks_scipy(ts)
    peaks_scipy_wct = find_peaks_wct(ts)
    # common points detected by different methods
    com1 = list(set(peaks_zscore.keys()).intersection(list(peaks_zcrossing.keys())))
    #com2 = set(peaks_scipy_wct.keys()).intersection(list(peaks_scipy.keys()))
    #common = list(set(com1).intersection(list(com2)))
    # keep those with highest SNR (90%)
    for t in com1 : 
        peaks[t] = peaks_zscore[t]
    # keeps only 80% of peaks with highet SNR
    snr = SNR(ts, peaks, w_size = 20)
    snr = {k:v for k, v in sorted(snr.items(), key=lambda item: item[1], reverse =True)}
    p_snr = 8*len(snr)//10
    key_snr_per = list(snr.keys())[0:p_snr]
    peaks_snr = {}
    for t in key_snr_per :
        peaks_snr[t] = peaks[t]
    return peaks_snr

## Scipy predefined function wrappers
def find_peaks_scipy(ts): 
    peaks  = {}
    t_peaks, _  = find_peaks(ts, distance=50)
    for t in  ts.index[t_peaks]:
        peaks[t] = ts[t]
    return peaks

def find_peaks_wct(ts): 
    peaks  = {}
    t_peaks = find_peaks_cwt(ts, widths= np.arange(1,20),noise_perc = 50)
    for t in ts.index[t_peaks]:
        peaks[t] = ts[t]
    return peaks

# Peak strength metrics
## Calculate local SNR for each peak in the time series
def SNR(ts, peaks, w_size):
    n_ts = len(ts)
    SNRs = {}
    for peak in peaks.keys() : 
        id_peak = list(ts.index).index(peak)
        # Sliding window around each peak
        portion = ts.index[max(id_peak - w_size//2, 0) : min(id_peak + w_size//2, n_ts)]
        SNRs[peak] = _signal_to_noise(ts[portion], axis=0,ddof=0)
    return SNRs

## Calculate SNR 
def _signal_to_noise(ts, axis=0, ddof = 0) :
     ts = np.asanyarray(ts)
     m = ts.mean(axis)
     sd = ts.std(axis=axis, ddof=ddof)
     return np.where(sd == 0, 0, m/sd)

# Zero-crossing: Variance of z-score function, where peak is calculated from some defined functions S1,S2...
def zero_crossing(ts, k, h, measure):  # 1<=h<=3 // measure ='S1' or 'S2'
    ts_n = len(ts)
    peaks = {}
    a = np.array(ts)
    m_prime, std_prime= np.array(ts), np.array(ts)
    for i in range(k, ts_n):
        if measure == 'S1': 
            a[i]= _S1(ts, k, i)
        else : 
            a[i] = _S2(ts, k, i)
    for i in range(k, ts_n):
        m_prime[i], std_prime[i] = np.mean(a[i-k+1: i+1]), np.std(a[i-k + 1: i+1])
    for i in range(k, ts_n):
        if a[i] > 0 and (a[i] - m_prime[i]) > h*std_prime[i]:
            peaks[i] = ts[i]
    # Keep only one point on the same lag-window of size k 
    pop_keys = []
    for i in peaks.keys(): 
        for j in peaks.keys(): 
            if abs(i-j) <= k and i!=j : 
                m = min(peaks[i], peaks[j])
                if peaks[i] == m:
                   pop_keys.append(i)
                else:
                   pop_keys.append(j)
    for k in set(pop_keys): 
        peaks.pop(k)
    peaks ={ts.index[k] : ts[k] for k in peaks.keys()}
    return peaks

## Zero-crossing algorithm
## You can use either S1 or S2 on the zero-crossing algorithm
def _S1(ts, k, i):
    max_before, max_after = ts[i] - np.min(ts[i-k-1: i]), ts[i] - np.min(ts[i+1: i+k+1])
    return (max_before + max_after)/2

def _S2(ts, k, i):
    mean_before, mean_after = ts[i] - np.mean(ts[i-k-1: i]),ts[i] - np.mean(ts[i+1: i+k+1])
    return (mean_before + mean_after)/2

## Z-score
def zscore(ts, lag, threshold, influence):
    """
       Calculate z-score for each point (after the lag), based on the means and std of its 'lag' past values.
       threshold : number of standard deviations from which we consider the point as a peak.
       Influence parameter : to decide how much past values can affect detection of future peaks.
    """
    signals = {}
    filteredY = np.array(ts)
    avgFilter = [0]*len(ts)
    stdFilter = [0]*len(ts)
    avgFilter[0:lag] = [np.mean(ts[0:lag]) for _ in range(lag)]
    stdFilter[0:lag] = [np.std(ts[0:lag]) for _ in range(lag)]
    for i in range(lag, len(ts)):
        if abs(ts[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:  # if it's a peak
            if ts[i] > avgFilter[i-1] :
                signals[ts.index[i]] = ts[i] # up peak
            # 'influence' control past values influence on next peaks 
            filteredY[i] = influence * ts[i] + (1 - influence) * filteredY[i-1]
        else:
            filteredY[i] = ts[i]
        # Average moving (filter the tiem series)
        avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
        # Std moving 
        stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
    # Create time series for output
    avgFilter = pd.Series(avgFilter, index=ts.index)
    stdFilter = pd.Series(stdFilter, index=ts.index)
    return dict(signals=signals, avgFilter=avgFilter, stdFilter=stdFilter)

## Calculate the prominence of each peak in a signal
def prominence(ts, peaks):
    id_peaks = []
    for peak in peaks.keys() :
        id_peaks.append(list(ts.index).index(peak))
    prominence = peak_prominences(ts, id_peaks)[0]
    proms = {}
    peaks_t = list(peaks.keys())
    for i in range(len(peaks_t)) : 
        proms[peaks_t[i]] = prominence[i]
    return proms

## LSTM method
### Prepare data: split time series to fit input of the vanilla_LSTM model
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# Time series gap handling
## Handle gaps
def ungap_periodic(df, colname, weight):
    ts = df[colname]
    ts_work = ts.copy()
    ts_periods = ts.copy().interpolate(method='time').round(2).fillna(0)
    periods = get_periods(ts_periods, min_strength=0.6, all=False)
    gaps = _detect_gaps(ts_work, colname)
    gaps_start, gaps_end = gaps['start'], gaps['end']
    if len(periods) > 0:
        period = periods[0]
        for i in range(len(gaps_start)):
            s = ts_work.index.get_loc(gaps_start[i])
            e = ts_work.index.get_loc(gaps_end[i]) + 1
            length = e - s
            times = math.ceil(length / period)
            if s > period:
                for _ in range(times):
                    l = min(period, length)
                    ts_work.iloc[s: s + l] = ts_work.iloc[s -
                                                          period: s - period + l].values
                    s += l
                    length -= l
    ts_work = ts_work.interpolate(method='time')
    ts_work = add_noise(ts_work, gaps, weight)
    return ts_work.round(3)

def ungap_knn(df, colname, n_neighbors, noise_weight=0):
    ts = df[colname]
    imputer = KNNImputer(n_neighbors=n_neighbors)
    ts_out = pd.Series(index=ts.index, data=imputer.fit_transform(ts.array.reshape(-1, 1)).reshape(-1))
    # Add noise if a noise weight is specified
    if noise_weight != 0:
        gaps = _detect_gaps(ts, colname)
        ts_out = add_noise(ts_out, gaps, noise_weight)
    return ts_out.round(3)

def ungap_mice(df, colname, standard_length, weight): 
    ts = df[colname]
    ts_work = ts.copy()
    gaps = _detect_gaps(ts_work, colname)
    gaps_start, gaps_end = gaps['start'], gaps['end']
    for k in range(len(gaps_start)):
        ts_gap = ts_work[:gaps_end[k]].copy()
        n_gaps = len(ts_work[gaps_start[k] : gaps_end[k]])
        mice_imputer = IterativeImputer()
        # The gap is smaller than or equal the standard length
        if n_gaps <= standard_length: 
            ts_pred = mice_imputer.fit_transform(np.array(ts_gap).reshape(-1,1))
            preds_part = pd.Series(ts_pred.reshape(len(ts_gap)), index=ts_gap.index)
            ts_work[gaps_start[k] : gaps_end[k]] = preds_part[gaps_start[k] : gaps_end[k]]
        # The gap is greater than the standard length. Divide it into parts.
        else:
            ts_gap_end = np.array(ts_gap)
            ts_gap_start = np.array(ts_work[:gaps_start[k]]).copy()
            n_start = len(ts_gap_start)
            n_parts = n_gaps // standard_length
            preds = []
            for i in range(n_parts - 1): 
                ts_s_gap = ts_gap_end[:n_start + (i + 1) * standard_length]
                ts_pred = mice_imputer.fit_transform(np.array(ts_s_gap).reshape(-1,1))
                preds_part = pd.Series(ts_pred.reshape(len(ts_s_gap)), index=range(len(ts_s_gap)))
                preds.extend(preds_part[n_start + i * standard_length : n_start + (i + 1) * standard_length])
                ts_gap_end[n_start + i * standard_length : n_start + (i + 1) * standard_length] = preds_part[n_start + i * standard_length : n_start + (i + 1) * standard_length]
            ts_gap_final = ts_gap_end
            ts_pred = mice_imputer.fit_transform(np.array(ts_gap_final).reshape(-1,1))
            pred = pd.Series(ts_pred.reshape(len(ts_gap_final)), index=range(len(ts_gap_final)))
            preds.extend(pred[n_start + (n_parts - 1) * standard_length - 1:])
            ts_work[gaps_start[k] : gaps_end[k]] = preds
    ts_work = add_noise(ts_work, gaps, weight)
    return ts_work.round(3)

## Add artificial noise
def add_noise(ts, gaps, weight):
    ts_work = ts.copy()
    gaps_start = list(gaps['start'])
    gaps_end = list(gaps['end'])
    for i in range(len(gaps_start)):
        gap = ts_work[gaps_start[i] : gaps_end[i]]
        norm = np.random.normal(np.mean(gap), np.std(gap), size = len(gap))
        ts_work[gaps_start[i] : gaps_end[i]] = weight * norm + (1 - weight) * gap
    return ts_work

def filter_outliers(ts, p1=0.25, p3=0.75, whisker_width=1.5):
    ts_work = ts.copy()
    q1, q3 = ts_work.quantile(p1), ts_work.quantile(p3)
    iqr = q3 - q1      
    ts_work[~ts_work.between(q1 - whisker_width * iqr, q3 + whisker_width * iqr, inclusive='both')] = np.NaN
    ts_work.interpolate(method='time').round(2)
    return ts_work

## Detect gaps
def _detect_gaps(ts, colname):
    ts_work = ts.copy()
    ts_work[ts_work < 0] = np.nan

    na_groups = ts_work.isna().cumsum()[ts_work.isna()]
    blocks = ts_work.diff().notna().cumsum()
    out = na_groups.index.to_frame().groupby(
        blocks)['timestamp'].agg(['min', 'max'])
    out.reset_index(inplace=True)
    out.rename({'min': 'start', 'max': 'end'}, axis=1, inplace=True)
    out.drop(colname, axis=1, inplace=True)
    return out

## Differencing method for time series 
def difference(ts):
    initial = ts[0]
    diff = ts.diff()[1:]
    # Keep initial for reconstruction purposes 
    return initial, diff

### Reconstruct time series from differenced one 
def inv_difference(initial, ts):
    ts_back = list(initial) + list(ts)
    for i in range(1, len(ts)): 
        ts_back[i] += ts_back[i-1] 
    return ts_back 

# Stationarity tests
def is_stationary(ts, sig_level=0.05, trend_stationary=False):
    reg = 'ct' if trend_stationary else 'c'

    adftest = sm.tsa.stattools.adfuller(ts, autolag='AIC')
    kpsstest = sm.tsa.stattools.kpss(ts, regression=reg)
    adf_pval, kpss_pval = adftest[1], kpsstest[1]
    #print(adf_pval, kpss_pval)
    return adf_pval < sig_level and kpss_pval >= sig_level

# Time series component strength
def _component_strength(component, residual):
    return max(0, 1 - np.var(residual) / np.var(component + residual))

# Seasonality strength
def seasonality_strength(decomp):
    return _component_strength(decomp.seasonal.values, decomp.resid.values)

# Trend strength
def trend_strength(decomp):
    return _component_strength(decomp.trend.values, decomp.resid.values)

# Seasonality periods detection
def get_periods(ts, min_strength=0.6, all=True):
    ts_work = ts.copy()
    periods = []

    while True:
        p, max_acf = None, -np.inf
        acf = sm.tsa.stattools.acf(ts_work, nlags=len(ts_work), fft=True)

        for i in range(1, len(acf) // 2):
            # Check i has not been previously detected
            if i in periods:
                continue

            # Check that i has a greater ACF value than the current candidate p
            if p is not None:
                if acf[i] < acf[p]:
                    continue

            # Assume i is a candidate
            candidate = True

            # Check that i is the greatest local maximum
            if acf[i - 1] < acf[i] and acf[i] > acf[i + 1] and acf[i] > max_acf:
                # Check that multiples of i are also local maxima
                for j in range(2, len(acf) // i):
                    if acf[i * j - 1] > acf[i * j] or acf[i * j] < acf[i * j + 1]:
                        candidate = False
                        break
                if candidate:
                    p = i
                    max_acf = acf[i]

        # No seasonality found, exit
        if not p:
            break
        else:
            stl = sm.tsa.STL(ts_work, period=p, seasonal_deg=1,
                             trend_deg=1, robust=False)
            decomp = stl.fit()
            strength = seasonality_strength(decomp)

            # Seasonality is too weak, exit
            if strength < min_strength:
                break

            # Add candidate to periods
            periods.append(p)

            if not all:
                break
            # De-seasonalize the time series
            ts_work -= decomp.seasonal

    return periods

# MAPE function
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Validation metrics
def forecast_report(forecast, ts_test):
    return round(r2_score(ts_test, forecast), 2), round(mean_absolute_error(ts_test, forecast), 2), round(mean_squared_error(ts_test, forecast), 2), round(mean_absolute_percentage_error(
        ts_test, forecast), 2), round(median_absolute_error(ts_test, forecast), 2)

# Plotting
def plot_fcast(train, test, fcast, xlabel, ylabel, title, legend=['Observed Past', 'Observed Future', 'Forecast'], filename='fig.png', ci=False, fcast_low=None, fcast_up=None):
    plt.figure()
    ax = train.plot(label=legend[0])
    if test is not None:
        test.plot(ax=ax, label=legend[1])
    if fcast is not None:
        fcast.plot(ax=ax, label=legend[2], alpha=0.75)
        plt.axvline(x=test.index[0], color='brown',
                    label='Present', linestyle='--')
        if ci:
            ax.fill_between(fcast.index,
                            fcast_low,
                            fcast_up, color='k', alpha=.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)

    plt.legend()
    plt.savefig(filename, bbox_inches='tight',
                pad_inches=0.5, transparent=True)
    plt.close()
