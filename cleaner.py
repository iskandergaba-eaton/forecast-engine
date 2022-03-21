import os
import numpy as np
import pandas as pd

from util import ungap
from sklearn.preprocessing import MultiLabelBinarizer

def get_colnames(filename, delimiter=",", empty_replacement=""):
    header = read_first_line(filename, ignore_header=False).replace("\n", "")
    colnames = header.split(delimiter)
    colnames = [empty_replacement if col == '' else col for col in colnames]
    return colnames

def get_time_range(filename, round='10T'):
        first_line = read_first_line(filename, ignore_header=True)
        last_line = read_last_line(filename)
        start_time = pd.to_datetime(first_line.split(',')[0], utc=True).round(round)
        end_time = pd.to_datetime(last_line.split(',')[0], utc=True).round(round)
        return start_time, end_time

def read_first_line(filename, ignore_header):
    with open(filename, 'r') as f:
        if ignore_header:
            try:  # Catch OSError in case of a one line file
                next(f) # Skip header
            except OSError:
                f.seek(0)
        return f.readline()

def read_last_line(filename):
    with open(filename, 'rb') as f:
        try:  # Catch OSError in case of a one line file 
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        return f.readline().decode()

def build_dc_state(on, off):
    current = []
    start_stop = {}
    for i in on.keys():
        start_list = on[i]
        current = current + start_list
        start_stop[i] = current
        for j in off.keys():
            if j < i:
                stop_list = off[j]
                current = [t for t in current if t not in stop_list]
                start_stop[j] = current
                off.pop(j, None)
            else:
                break
    mlb = MultiLabelBinarizer()
    servers_state = pd.DataFrame(mlb.fit_transform(start_stop.values()),
                   columns=mlb.classes_,
                   index=start_stop.keys()).sort_index()
    servers_state.index.name = 'timestamp'
    return servers_state

# Ungapping strategies
STRAT_UNGAP = 0
STRAT_INTERPOLATE = 1
STRAT_GAPS = 2

# Initial variables
root_dirty, root_clean = '../data/dirty', '../data/clean'
versions = ['20-12-2021', '12-01-2022', '19-02-2022', '16-03-2022']
server_on, server_off = {}, {}
start_times, end_times = {}, {}
files_dirty = []

# Choose ungapping strategy
strategy = STRAT_UNGAP

# Pre-processing
for path, subdirs, files in os.walk(os.path.join(root_dirty, versions[-1])):
    for name in files:
        filename = os.path.join(path, name)
        files_dirty.append(filename)

        # Prepare data folders
        savename = filename.replace(root_dirty, root_clean)
        save_dir, server = os.path.split(savename)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dc = os.path.split(save_dir)[1]
        if dc not in start_times:
            start_times[dc] = {}
            end_times[dc] = {}
        
        # Store time ranges for each server in each datacenter
        start_t, end_t = get_time_range(filename)
        start_times[dc][start_t] = 1 if start_t not in start_times[dc] else start_times[dc][start_t] + 1
        end_times[dc][end_t] = 1 if end_t not in end_times[dc] else end_times[dc][end_t] + 1

        if dc not in server_on:
            server_on[dc] = {}
            server_off[dc] = {}

        # Store timestamps where servers start or stop
        server_on[dc][start_t] = [server] if start_t not in server_on[dc] else server_on[dc][start_t] + [server]
        server_off[dc][end_t] = [server] if end_t not in server_off[dc] else server_off[dc][end_t] + [server]


# Sort the list of files for debug purposes
files_dirty.sort()

# Find the most dominant time range for each datacenter
min_timestamp, max_timestamp = {}, {}
for dc in start_times:
    min_timestamp[dc] = max(start_times[dc], key=start_times[dc].get)
    max_timestamp[dc] = max(end_times[dc], key=end_times[dc].get)

# Build a dataframe for server states for every datacenter
for dc in server_on:
    savename = os.path.join(root_clean, versions[-1], '{}.csv'.format(dc))
    df = build_dc_state(server_on[dc], server_off[dc])
    df.to_csv(savename)

# Start cleaning
for filename in files_dirty:

    # get column names list
    colnames = get_colnames(filename, empty_replacement="timestamp")

    # Ignore VMs
    if 'Host' in colnames:
        continue

    # Load the data
    df = pd.read_csv(filename, index_col=0, header=0, names=colnames,
    usecols=["timestamp", "power"], parse_dates=True,
    date_parser=lambda col: pd.to_datetime(col, utc=True))

    # Round the index to the nearest 10 minutes
    freq = '10T'
    df.index = df.index.round(freq)

    # Preparing save paths
    savename = filename.replace(root_dirty, root_clean)
    save_dir, server = os.path.split(savename)
    dc = os.path.split(save_dir)[1]

    # Ignore server if the data range is not long enough and power consumption has not been recorded
    if df.index[0] > min_timestamp[dc] or df.index[-1] < max_timestamp[dc] or df['power'].max() == 0:
        print(server, 'skipped.')
        continue
    else:
        df = df[min_timestamp[dc]:max_timestamp[dc]]
    
    # Start cleaning
    print('{} cleaning started...'.format(server))

    # Rename columns
    if 'cpux100' in df.columns:
        df['cpux100'] /= 100

    # Resample data
    ts_power = df['power'].copy()
    df = df.resample(freq).agg(np.mean)
    df['power_max'] = ts_power.resample(freq).agg(np.max)
    df.index.freq = freq

    # Apply the chosen ungapping strategy
    if strategy == STRAT_UNGAP:
        df['power'] = ungap(df, 'power').round(2)
        df['power_max'] = ungap(df, 'power_max').round(2)
    elif strategy == STRAT_INTERPOLATE:
        df['power'] = df['power'].interpolate(method='time').round(2)
        df['power_max'] = df['power_max'].interpolate(method='time').round(2)

    # Save clean data
    df.to_csv(savename)
    print('{} cleaning done.'.format(server))
