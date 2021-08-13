import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 18, 8


from engine import ForecastEngine
from util import load_data, fill_gaps


if __name__ == "__main__":

    DCs = ['usltcsvcenter2', 'usltcsvcenter3',
           'usltccnvcenter4', 'usltccnvcenter5']

    root, filename, dc = '../data', 'fig.png', DCs[3]

    horizon_key = ForecastEngine.HORIZON_MID

    oracle = ForecastEngine(root=root)
    fcast, fcast_low, fcast_up = oracle.forecast(dc=dc, approach=ForecastEngine.APPROACH_HYBRID,
                                                 granularity=ForecastEngine.GRANULARITY_DC, horizon=horizon_key)

    df = load_data(root, dc, freq=fcast.index.freq)
    ts_power = df['power']
    ts_power.index.freq = fcast.index.freq

    horizon, size, split = ForecastEngine._horizons[horizon_key], ForecastEngine._storage_sizes[horizon_key], len(ts_power) - ForecastEngine._splits[horizon_key]

    ts_power_train = ts_power[:split] if horizon_key == ForecastEngine.HORIZON_LONG else ts_power[split-size:split]
    ts_power_test = ts_power.copy()[split:split + horizon]

    ax = ts_power_train.plot(label='Observed Past')

    ts_power_test.plot(ax=ax, label='Observed Future')

    # Simulate the missing data
    ts_power_filled = fill_gaps(ts_power) if horizon_key == ForecastEngine.HORIZON_LONG else fill_gaps(ts_power)[split-size:split+horizon]
    ts_power_filled -= ts_power.fillna(0)
    ts_power_filled[ts_power_filled == 0] = np.nan
    ts_power_filled.plot(ax=ax, label='Simulated Data')

    print(fcast)

    fcast.plot(ax=ax, label='Forecast', alpha=0.75, color='teal')
    ax.fill_between(fcast.index, fcast_low, fcast_up, color='k', alpha=.25)
    plt.axvline(x=fcast.index[0], color='brown',
                label='Present', linestyle='--')

    ax.set_xlabel('Time')
    ax.set_ylabel('Power Comsuption (Watts)')
    plt.title('Power Comsumption Forecast')

    plt.legend()
    plt.savefig(filename, bbox_inches='tight',
                pad_inches=0.5, transparent=True)
    plt.close()
