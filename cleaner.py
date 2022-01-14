import os
import pandas as pd

from util import fill_gaps

root_dirty, root_clean = '../data/dirty', '../data/clean'
versions = ['20-12-2021', '12-01-2022']
min_timestamp, max_timestamp = {}, {}

files_dirty = []
for path, subdirs, files in os.walk(os.path.join(root_dirty, versions[-1])):
    for name in files:
        files_dirty.append(os.path.join(path, name))

for filename in files_dirty:

    # Prepare data folders
    savename = filename.replace(root_dirty, root_clean)
    save_dir, server = os.path.split(savename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dc = os.path.split(save_dir)[1]

    # Load the data
    df = pd.read_csv(filename, error_bad_lines=True)
    # Rename columns
    df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True),

    # Convert 'timestamp` to datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Round to nearest minute
    df = df.groupby(df['timestamp'].dt.round('10 min')).mean().reset_index()

    # Ensuring that we only take time series that are long enough into consideration
    min_t, max_t = min(df['timestamp']), max(df['timestamp'])
    min_timestamp[dc] = min(min_t, min_timestamp[dc]) if dc in min_timestamp else min_t
    max_timestamp[dc] = max(max_t, max_timestamp[dc]) if dc in max_timestamp else max_t
    if dc in min_timestamp:
        min_timestamp[dc] = max(min_t, min_timestamp[dc]) if str(
            min_timestamp[dc] - min_t)[0] == '0' else min(min_t, min_timestamp[dc])
    else:
        min_timestamp[dc] = min_t
    if dc in max_timestamp:
        max_timestamp[dc] = min(max_t, max_timestamp[dc]) if str(
            max_timestamp[dc] - max_t)[0] == '0' else max(max_t, max_timestamp[dc])
    else:
        max_timestamp[dc] = max_t

for filename in files_dirty:

    # Prepare data folders
    savename = filename.replace(root_dirty, root_clean)
    save_dir, server = os.path.split(savename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load the data
    df = pd.read_csv(filename, error_bad_lines=True)

    # Ignore server if all power values are zeros
    if (df['power'] == 0).all():
        print(server, 'skipped.')
        continue

    # Ignore VMs
    if 'Host' in df.columns:
        continue

    print('Cleaning {}...'.format(server))

    # Rename columns
    df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True),
    if 'cpux100' in df.columns:
        df['cpux100'] /= 100

    # Convert 'timestamp` to datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Round to nearest minute
    df = df.groupby(df['timestamp'].dt.round('10 min')).mean().reset_index()

    # Set the index
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # Fill gaps via interpolation
    df = df.resample('10T').mean().round(2)
    df['power'] = fill_gaps(df['power'])

    # Ignore server if the recording length is not enough
    dc = os.path.split(save_dir)[1]
    if df.index[0] > min_timestamp[dc] or df.index[-1] < max_timestamp[dc] or df['power'].max() == 0:
        print(server, 'skipped.')
        continue
    else:
        idx = min(df.index[-1], max_timestamp[dc])
        df = df[:idx]

    # Save clean data
    df.to_csv(savename)
    print('Done.')
