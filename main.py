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

       root, dc = '../data/clean/{}'.format(versions[-1]), DCs[0]

       horizon_key = ForecastEngine.HORIZON_90
       horizon, size = ForecastEngine._horizons[horizon_key], ForecastEngine._storage_sizes[horizon_key]
       oracle = ForecastEngine(root=root)

       # Mean power consumption forecast
       fcast, fcast_low, fcast_up = oracle.forecast(dc=dc, series='power', approach=ForecastEngine.APPROACH_REGULAR,
                                          granularity=ForecastEngine.GRANULARITY_SMART, horizon=horizon_key)
       df = load_data(root, dc, freq=fcast.index.freq)
       ts_power = df['power']
       ts_power.index.freq = fcast.index.freq

       split = ts_power.index[-1] - ForecastEngine._test_sizes[horizon_key]
       ts_power_train = ts_power[:split] if horizon_key == ForecastEngine.HORIZON_180 else ts_power[split - size:split]
       ts_power_test = ts_power.copy()[split:]
       
       figname = '.results/{}.png'.format(dc)
       ax = ts_power_train.plot(label='Observed Past')
       ts_power_test.plot(ax=ax, label='Observed Future')
       fcast.plot(ax=ax, label='Forecast', alpha=0.75, color='teal')
       ax.fill_between(fcast.index, fcast_low, fcast_up, label='Confidence Interval', color='k', alpha=.25)
       plt.axvline(x=fcast.index[0], color='brown',label='Present', linestyle='--')
       ax.set_xlabel('Time')
       ax.set_ylabel('Power Comsuption (Watts)')
       plt.title('Mean Power Comsumption Forecast')
       plt.legend()
       plt.savefig(figname, bbox_inches='tight', pad_inches=0.5, transparent=True)
       plt.close()

       # Peak power consumption forecast
       fcast_peak, fcast_peak_low, fcast_peak_up = oracle.forecast(dc=dc, series='power_max', approach=ForecastEngine.APPROACH_REGULAR,
                                          granularity=ForecastEngine.GRANULARITY_SMART, horizon=horizon_key, agg_func=np.max)
       df_peak = load_data(root, dc, agg_func=np.max, freq=fcast.index.freq)
       ts_power_peak = df_peak['power_max']
       ts_power.index.freq = fcast.index.freq
       ts_power_peak.index.freq = fcast_peak.index.freq

       split_peak = ts_power_peak.index[-1] - ForecastEngine._test_sizes[horizon_key]
       ts_power_peak_train = ts_power_peak[:split_peak] if horizon_key == ForecastEngine.HORIZON_180 else ts_power_peak[split_peak - size:split_peak]
       ts_power_peak_test = ts_power_peak.copy()[split_peak:]

       figname_peak = '.results/{}-peak.png'.format(dc)
       ax = ts_power_peak_train.plot(label='Observed Past')
       ts_power_peak_test.plot(ax=ax, label='Observed Future')
       fcast_peak.plot(ax=ax, label='Forecast', alpha=0.75, color='teal')
       ax.fill_between(fcast_peak.index, fcast_peak_low, fcast_peak_up, label='Confidence Interval', color='k', alpha=.25)
       plt.axvline(x=fcast_peak.index[0], color='brown',label='Present', linestyle='--')
       ax.set_xlabel('Time')
       ax.set_ylabel('Power Comsuption (Watts)')
       plt.title('Peak Power Comsumption Forecast')
       plt.legend()
       plt.savefig(figname_peak, bbox_inches='tight', pad_inches=0.5, transparent=True)
       plt.close()
