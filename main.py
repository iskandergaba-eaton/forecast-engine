import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 18, 8

from engine import ForecastEngine
from util import load_data


if __name__ == "__main__":

    DCs = ['usltcsvcenter2', 'usltcsvcenter3',
           'usltccnvcenter4', 'usltccnvcenter5', 'simtcsvc01']
    versions = ['20-12-2021', '12-01-2022']

    root, dc = '../data/clean/{0}'.format(versions[-1]), DCs[2]
    figname = '.results/{0}.png'.format(dc)

    horizon_key = ForecastEngine.HORIZON_180

    oracle = ForecastEngine(root=root)
    fcast, fcast_low, fcast_up = oracle.forecast(dc=dc, series='power', approach=ForecastEngine.APPROACH_REGULAR,
                                                 granularity=ForecastEngine.GRANULARITY_SMART, horizon=horizon_key)
    fcast_peaks, _, _ = oracle.forecast(dc=dc, series='power_max', approach=ForecastEngine.APPROACH_REGULAR,
                                                 granularity=ForecastEngine.GRANULARITY_SMART, horizon=horizon_key, agg_func=np.max)

    df = load_data(root, dc, freq=fcast.index.freq)
    ts_power = df['power']
    ts_power.index.freq = fcast.index.freq

    horizon, size, split = ForecastEngine._horizons[horizon_key], ForecastEngine._storage_sizes[horizon_key], ts_power.index[-1] - ForecastEngine._test_sizes[horizon_key]

    ts_power_train = ts_power[:split] if horizon_key == ForecastEngine.HORIZON_180 else ts_power[split - size:split]
    ts_power_test = ts_power.copy()[split:]

    ax = ts_power_train.plot(label='Observed Past')

    ts_power_test.plot(ax=ax, label='Observed Future', color='teal')

    fcast.plot(ax=ax, label='Forecast', alpha=0.75, color='yellow')
    fcast_peaks.plot(ax=ax, label='Peaks Forecast', alpha=0.75, color='red')
    ax.fill_between(fcast.index, fcast_low, fcast_up, label='Confidence Interval', color='k', alpha=.25)
    plt.axvline(x=fcast_peaks.index[0], color='brown',
                label='Present', linestyle='--')

    ax.set_xlabel('Time')
    ax.set_ylabel('Power Comsuption (Watts)')
    plt.title('Power Comsumption Forecast')

    plt.legend()
    plt.savefig(figname, bbox_inches='tight',
                pad_inches=0.5, transparent=True)
    plt.close()
