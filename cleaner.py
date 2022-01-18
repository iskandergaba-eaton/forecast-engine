import os
import numpy as np
import pandas as pd

from util import ungap

def get_time_range(filename):
        first_line = read_first_line(filename, ignore_header=True)
        last_line = read_last_line(filename)
        start_time = pd.to_datetime(first_line.split(',')[0], utc=True).round('10T')
        end_time = pd.to_datetime(last_line.split(',')[0], utc=True).round('10T')
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

# Global variables
root_dirty, root_clean = '../data/dirty', '../data/clean'
versions = ['20-12-2021', '12-01-2022']
start_times, end_times = {}, {}
files_dirty = []

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
        start_t, end_t = get_time_range(filename, ignore_header=True)
        start_times[dc][start_t] = 1 if start_t not in start_times[dc] else start_times[dc][start_t] + 1
        end_times[dc][end_t] = 1 if end_t not in end_times[dc] else end_times[dc][end_t] + 1

# Find the most dominant time range for each datacenter
min_timestamp, max_timestamp = {}, {}
for dc in start_times:
    min_timestamp[dc] = max(start_times[dc], key=start_times[dc].get)
    max_timestamp[dc] = max(end_times[dc], key=end_times[dc].get)

# Start cleaning
for filename in files_dirty:
    # Load the data
    df = pd.read_csv(filename, error_bad_lines=True)

    # Ignore VMs
    if 'Host' in df.columns:
        continue

    # Ignore server if the data range is not long enough and power consumption has not been recorded
    dc = os.path.split(save_dir)[1]
    if df.index[0] > min_timestamp[dc] or df.index[-1] < max_timestamp[dc] or df['power'].max() == 0:
        print(server, 'skipped.')
        continue
    else:
        df = df[min_timestamp[dc]:max_timestamp[dc]]
    
    # Start cleaning
    print('{} cleaning started...'.format(server))

    # Rename columns
    df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True),
    if 'cpux100' in df.columns:
        df['cpux100'] /= 100

    # Convert `timestamp` to a datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Set the index and round to nearest minute
    freq = '10T'
    df.set_index('timestamp', inplace=True)
    df.index = df.index.round(freq)
    ts_power = df['power'].copy()
    df = df.resample(freq).agg(np.mean).round(2)
    df['power_max'] = ts_power.resample(freq).agg(np.max).round(2)
    df.index.freq = freq

    # Fill gaps via interpolation
    df['power'] = ungap(df, 'power')
    df['power_max'] = ungap(df, 'power_max')

    # Save clean data
    df.to_csv(savename)
    print('{} cleaning done.'.format(server))
