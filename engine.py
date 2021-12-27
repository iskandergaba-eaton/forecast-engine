from datetime import timedelta
import os

import util
import pandas as pd
import statsmodels.tsa as sm

class ForecastEngine:

    # Forecasting approaches
    APPROACH_REGULAR = 0
    APPROACH_HYBRID = 1

    # Granularity levels
    GRANULARITY_SERVER = 0
    GRANULARITY_SMART = 1
    GRANULARITY_DC = 2

    # Forecast horizons
    HORIZON_SHORT = 2
    HORIZON_MID = 1
    HORIZON_LONG = 0

    # Parameter dictionaries
    _horizons = {0: timedelta(days=180), 1: timedelta(days=30), 2: timedelta(days=7)}
    _test_sizes = {0: timedelta(days=90), 1: timedelta(days=30), 2: timedelta(days=7)}
    _freqs = {0: '2H', 1: 'H', 2: '10min'}
    _storage_sizes = {0: timedelta(days=180), 1: timedelta(days=90), 2: timedelta(days=7)}

    def __init__(self, root):
        self.root = root
    
    def _forecast(self, ts, start_future, end_future, alpha=0.05):
        # Get seasonality periods
        ts_periods = ts.copy().interpolate(method='time').round(2).fillna(0)
        periods = util.get_periods(ts_periods, min_strength=0.5, all=False)
        print('Periods:', periods)

        model = None
        if len(periods) > 0:
            model = sm.forecasting.stl.STLForecast(
                ts, sm.arima.model.ARIMA, model_kwargs=dict(order=(1,1,1)), period=periods[0])
        else:
            model = sm.arima.model.ARIMA(ts, order=(1, 1, 1))

        result = model.fit()
        pred = result.get_prediction(start=start_future, end=end_future).summary_frame(alpha=alpha)
        return pred

    def _server(self, dc, horizon, save_dir='.results'):
        h = self._horizons[horizon]
        f = self._freqs[horizon]

        # Create save directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filenames = util.get_servers(self.root, dc)

        fcast = fcast_low = fcast_up = None
        fcast_agg = fcast_agg_low = fcast_agg_up = None

        for name in filenames:
            # Loading data
            df = util.load_file(name, freq=f)

            # Preprocessing
            ts_power = df['power']
            ts_power.index.freq = f
            split = ts_power.index[-1] - self._test_sizes[horizon]

            ts_power_train = ts_power[:split]
            ts_power_test = ts_power[split:]

            # Forecasting
            pred = self._forecast(
                ts_power_train, start_future=ts_power_test.index[0], end_future=ts_power_test.index[0] + h, alpha=0.05)
            fcast, fcast_low, fcast_up = pred['mean'], pred['mean_ci_lower'], pred['mean_ci_upper']

            # Aggregate forecasts
            if fcast_agg is None:
                fcast_agg, fcast_agg_low, fcast_agg_up = fcast, fcast_low, fcast_up
            else:
                fcast_agg = fcast_agg.add(fcast)
                fcast_agg_low = fcast_agg_low.add(fcast_low)
                fcast_agg_up = fcast_agg_up.add(fcast_up)

        return fcast_agg, fcast_agg_low, fcast_agg_up

    def _smart(self, dc, horizon, save_dir='.results'):
        h = self._horizons[horizon]
        f = self._freqs[horizon]

        # Create save directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filenames = util.get_servers(self.root, dc)

        groups = util.group_servers(filenames, h, f)

        fcast_agg = fcast_agg_low = fcast_agg_up = None

        for g in groups:
            fcast = fcast_low = fcast_up = None
            ts_power = groups[g]

            # Preprocessing
            ts_power.index.freq = f
            split = ts_power.index[-1] - self._test_sizes[horizon]

            ts_power_train = ts_power[:split]
            ts_power_test = ts_power[split:]

            # Forecasting
            pred = self._forecast(
                ts_power_train, start_future=ts_power_test.index[0], end_future=ts_power_test.index[0] + h, alpha=0.05)
            fcast, fcast_low, fcast_up = pred['mean'], pred['mean_ci_lower'], pred['mean_ci_upper']

            # Aggregate forecasts
            if fcast_agg is None:
                fcast_agg, fcast_agg_low, fcast_agg_up = fcast, fcast_low, fcast_up
            else:
                fcast_agg = fcast_agg.add(fcast)
                fcast_agg_low = fcast_agg_low.add(fcast_low)
                fcast_agg_up = fcast_agg_up.add(fcast_up)

        return fcast_agg, fcast_agg_low, fcast_agg_up

    def _dc(self, dc, horizon, save_dir='.results'):
        h = self._horizons[horizon]
        f = self._freqs[horizon]

        # Create save directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Loading data
        df = util.load_data(self.root, dc, freq=f)

        # Preprocessing
        ts_power = df['power']
        ts_power.index.freq = f

        # Train-test split
        split = ts_power.index[-1] - self._test_sizes[horizon]
        ts_power_train = ts_power[:split]
        ts_power_test = ts_power[split:]

        # Forecasting
        pred = self._forecast(
            ts_power_train, start_future=ts_power_test.index[0], end_future=ts_power_test.index[0] + h, alpha=0.05)

        fcast, fcast_low, fcast_up = pred['mean'], pred['mean_ci_lower'], pred['mean_ci_upper']
        return fcast, fcast_low, fcast_up

    def _server_hybrid(self, dc, horizon, save_dir='.results'):
        # Create save directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        df = pd.DataFrame()

        filenames = util.get_servers(self.root, dc)
        fcast_acc, fcast_acc_low, fcast_acc_up = None, None, None

        for name in filenames:

            ts_power_old, fcast_old, fcast_old_low, fcast_old_up = None, None, None, None

            for i in range(horizon + 1):
                h = self._horizons[i]
                f = self._freqs[i]
                size = self._storage_sizes[i]

                # Loading data
                df = util.load_file(name, freq=f)

                # Preprocessing
                ts_power = df['power']
                ts_power.index.freq = f

                # Train-test split
                split = ts_power.index[-1] - self._test_sizes[i]
                ts_power_train = ts_power.copy()[:split] if ts_power_old is None else ts_power.copy()[split-size:split]
                ts_power_test = ts_power.copy()[split:split+h]

                # Preprocessing
                if ts_power_old is not None:
                    ts_power_train_old = ts_power_old.resample(f).mean().interpolate('time').reindex(index=ts_power.index, method='nearest')
                    ts_power_train_old = ts_power_train_old[ts_power_train.index]
                    ts_power_train -= ts_power_train_old

                # Storing older time series
                ts_power_old = ts_power

                # Forecasting
                pred = self._forecast(
                    ts_power_train, start_future=ts_power_test.index[0], end_future=ts_power_test.index[0] + h, alpha=0.05)
                fcast, fcast_low, fcast_up = pred['mean'], pred['mean_ci_lower'], pred['mean_ci_upper']

                # Processing forecast
                if fcast_old is not None:
                    fcast_old = fcast_old.copy().resample(f).mean().interpolate(
                        'time').reindex(index=fcast.index, method='nearest')
                    fcast_old_low = fcast_old_low.copy().resample(f).mean().interpolate(
                        'time').reindex(index=fcast.index, method='nearest')
                    fcast_old_up = fcast_old_up.copy().resample(f).mean().interpolate(
                        'time').reindex(index=fcast.index, method='nearest')
                    fcast += fcast_old
                    fcast_low += fcast_old_low
                    fcast_up += fcast_old_up

                # Aggregate forecasts
                if fcast_acc is not None:
                    # Reindex for different lengths of time series
                    if len(fcast_acc) < len(fcast):
                        fcast_acc = fcast_acc.reindex(index=fcast.index, fill_value=0)
                        fcast_acc_low = fcast_acc_low.reindex(index=fcast_low.index, fill_value=0)
                        fcast_acc_up = fcast_acc_up.reindex(index=fcast_up.index, fill_value=0)
                    else:
                        fcast = fcast.reindex(index=fcast_acc.index, fill_value=0)
                        fcast_low = fcast_low.reindex(index=fcast_acc_low.index, fill_value=0)
                        fcast_up = fcast_up.reindex(index=fcast_acc_up.index, fill_value=0)
                    fcast_acc = fcast_acc.add(fcast)
                    fcast_acc_low = fcast_acc_low.add(fcast_low)
                    fcast_acc_up = fcast_acc_up.add(fcast_up)
                else:
                    fcast_acc, fcast_acc_low, fcast_acc_up = fcast, fcast_low, fcast_up

                # Storing old forecast
                fcast_old = fcast.copy()
                fcast_old_low = fcast_low.copy()
                fcast_old_up = fcast_up.copy()

        return fcast_acc, fcast_acc_low, fcast_acc_up

    def _smart_hybrid(self, dc, horizon, save_dir='.results'):
        # Create save directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        configs = {}
        filenames = util.get_servers(self.root, dc)

        # Longest horizon to be used for grouping
        h = self._horizons[self.HORIZON_LONG]
        f = self._freqs[self.HORIZON_LONG]
        groups = util.group_servers(filenames, h, f, load=False)

        ts_power_old, fcast_old, fcast_old_low , fcast_old_up = {}, {}, {}, {}

        for i in range(horizon + 1):
            h = self._horizons[i]
            f = self._freqs[i]
            size = self._storage_sizes[i]

            for g in groups:
                # Loading data
                filenames = groups[g]
                df = util.load_files(filenames, f)

                # Preprocessing
                ts_power = df['power']
                ts_power.index.freq = f

                # Train-test split
                split = ts_power.index[-1] - self._test_sizes[i]
                ts_power_train = ts_power.copy()[:split] if ts_power_old is None else ts_power.copy()[split-size:split]
                ts_power_test = ts_power.copy()[split:split+h]

                # Preprocessing
                if g in fcast_old:
                    ts_power_train_old = ts_power_old[g].resample(f).mean().interpolate('time').reindex(index=ts_power.index, method='nearest')
                    ts_power_train_old = ts_power_train_old[ts_power_train.index]
                    ts_power_train -= ts_power_train_old

                # Storing older time series
                ts_power_old[g] = ts_power.copy()

                # Forecasting
                pred = self._forecast(
                    ts_power_train, start_future=ts_power_test.index[0], end_future=ts_power_test.index[0] + h, alpha=0.05)
                fcast, fcast_low, fcast_up = pred['mean'], pred['mean_ci_lower'], pred['mean_ci_upper']

                # Processing forecast
                if g in fcast_old:
                    fcast_old[g] = fcast_old[g].copy().resample(f).mean().interpolate(
                        'time').reindex(index=fcast.index, method='nearest')
                    fcast_old_low[g] = fcast_old_low[g].copy().resample(f).mean().interpolate(
                        'time').reindex(index=fcast.index, method='nearest')
                    fcast_old_up[g] = fcast_old_up[g].copy().resample(f).mean().interpolate(
                        'time').reindex(index=fcast.index, method='nearest')
                    fcast = fcast.add(fcast_old[g])
                    fcast_low = fcast_low.add(fcast_old_low[g])
                    fcast_up = fcast_up.add(fcast_old_up[g])

                # Aggregate forecasts
                if i not in configs:
                    configs[i] = (fcast, fcast_low, fcast_up)
                else:
                    fcast_acc, fcast_acc_low, fcast_acc_up = configs[i]

                    # Reindex for different lengths of time series
                    if len(fcast_acc) < len(fcast):
                        fcast_acc = fcast_acc.reindex(index=fcast.index, fill_value=0)
                        fcast_acc_low = fcast_acc_low.reindex(index=fcast_low.index, fill_value=0)
                        fcast_acc_up = fcast_acc_up.reindex(index=fcast_up.index, fill_value=0)
                    else:
                        fcast = fcast.reindex(index=fcast_acc.index, fill_value=0)
                        fcast_low = fcast_low.reindex(index=fcast_acc_low.index, fill_value=0)
                        fcast_up = fcast_up.reindex(index=fcast_acc_up.index, fill_value=0)

                    fcast_acc = fcast_acc.add(fcast)
                    fcast_acc_low = fcast_acc_low.add(fcast_low)
                    fcast_acc_up = fcast_acc_up.add(fcast_up)

                # Storing old forecast
                fcast_old[g] = fcast.copy()
                fcast_old_low[g] = fcast_low.copy()
                fcast_old_up[g] = fcast_up.copy()

        return configs[horizon]

    def _dc_hybrid(self, dc, horizon, save_dir='.results'):
        # Create save directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        df = pd.DataFrame()
        ts_power_old, fcast_old, fcast_old_low, fcast_old_up = None, None, None, None
        for i in range(horizon + 1):
            h = self._horizons[i]
            f = self._freqs[i]
            size = self._storage_sizes[i]

            # Loading data
            df = util.load_data(self.root, dc, freq=f)
            ts_power = df['power']
            ts_power.index.freq = f
            ts_power = util.fill_gaps(ts_power)

            # Split into train-test
            split = ts_power.index[-1] - self._test_sizes[i]
            ts_power_train = ts_power.copy()[:split] if ts_power_old is None else ts_power.copy()[split-size:split]
            ts_power_test = ts_power.copy()[split:split+h]

            # Train-test split
            split = ts_power.index[-1] - self._test_sizes[i]
            ts_power_train = ts_power.copy()[:split] if ts_power_old is None else ts_power.copy()[split-size:split]
            ts_power_test = ts_power.copy()[split:split+h]

            # Preprocessing
            if ts_power_old is not None:
                ts_power_train_old = ts_power_old.resample(f).mean().interpolate('time').reindex(index=ts_power.index, method='nearest')
                ts_power_train_old = ts_power_train_old[ts_power_train.index]
                ts_power_train -= ts_power_train_old

            # Storing older time series
            ts_power_old = ts_power.copy()
            

            # Forecasting
            pred = self._forecast(
                ts_power_train, start_future=ts_power_test.index[0], end_future=ts_power_test.index[0] + h, alpha=0.05)
            fcast, fcast_low, fcast_up = pred['mean'], pred['mean_ci_lower'], pred['mean_ci_upper']

            # Handling NaN edge case
            if (fcast_low.isna()).all():
                fcast_low = fcast.copy()
                fcast_up = fcast.copy()
            else:
                fcast_low.fillna(0, inplace=True)
                fcast_up.fillna(0, inplace=True)

            if fcast_old is not None:
                fcast_old = fcast_old.resample(f, closed='right', label='right').interpolate(
                    'time').reindex(index=fcast.index, method='nearest')
                fcast_old_low = fcast_old_low.resample(f, closed='right', label='right').interpolate(
                    'time').reindex(index=fcast.index, method='nearest')
                fcast_old_up = fcast_old_up.resample(f, closed='right', label='right').interpolate(
                    'time').reindex(index=fcast.index, method='nearest')
            
                fcast += fcast_old
                fcast_low += fcast_old_low
                fcast_up += fcast_old_up

            # Storing older forecast
            fcast_old = fcast.copy()
            fcast_old_low = fcast_low.copy()
            fcast_old_up = fcast_up.copy()
        
        return fcast, fcast_low, fcast_up

    def forecast(self, dc, approach, granularity, horizon):
        if approach == self.APPROACH_REGULAR:
            if granularity == self.GRANULARITY_SERVER:
                return self._server(dc, horizon)
            elif granularity == self.GRANULARITY_SMART:
                return self._smart(dc, horizon)
            else:
                return self._dc(dc, horizon)
        else:
            if granularity == self.GRANULARITY_SERVER:
                return self._server_hybrid(dc, horizon)
            elif granularity == self.GRANULARITY_SMART:
                return self._smart_hybrid(dc, horizon)
            else:
                return self._dc_hybrid(dc, horizon)
