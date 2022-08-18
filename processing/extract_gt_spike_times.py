import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from argparse import ArgumentParser, RawTextHelpFormatter
import pandas as pd
import warnings


def load_data_summary(file):
    df = pd.read_excel(file)
    return df


def get_spike_times_file(path):
    files = os.listdir(path)
    for file in files:
        if file.endswith('spike_samples.npy'):
            return file
    warnings.warn('No spike times file found!')
    return None


def load_data(path, cell, chan):
    # returns all data needed including the one channel from the NP recording
    path = os.path.join(path, cell)
    data = {}
    # load meta information
    meta = np.load(os.path.join(
        path, f'{cell}_expt_meta.npy'), allow_pickle=True)
    meta = meta.reshape(1, -1)
    patch_meta, npx_meta = meta[0][0]['patch'], meta[0][0]['npx']

    # load patch data
    data['patch'] = np.fromfile(os.path.join(
        path, f'{cell}_patch_ch1.bin'), dtype=patch_meta[-1])

    # load patch sync data
    data['patch_sync'] = np.fromfile(os.path.join(
        path, f'{cell}_patch_sync.bin'), dtype=patch_meta[-1])

    # load npx sync data
    data['npx_sync'] = np.fromfile(os.path.join(
        path, f'{cell}_npx_sync.bin'), dtype=npx_meta[-1])

    # load spike times
    spike_times_file = get_spike_times_file(path)
    if spike_times_file is not None:
        data['spike_times'] = np.load(os.path.join(
            path, spike_times_file), allow_pickle=True)

    # load npx channel data
    # unnecessarily loads all the data. TODO: load only the channel of interest
    # tmp = np.fromfile(os.path.join(
    #     path, f'{cell}_npx_raw.bin'), dtype=npx_meta[-1])
    # stride = npx_meta[0][0]
    # chan = int(chan)
    # data['npx_chan'] = tmp[chan::stride]

    data['folder'] = cell
    return data


def convert_patch_to_np_times(data):
    # find sync pulse times, fit a linear model to map to np times, apply model to convert spike times
    params = {'height': 0.5, 'distance': 5000}
    patch_peaks, _ = find_peaks(data['patch_sync'], **params)
    npx_peaks, _ = find_peaks(data['npx_sync'], **params)
    p = np.polyfit(patch_peaks, npx_peaks, deg=1)
    # linear mapping from spike_times on patch clock to npx clock
    spike_times_npx = (p[0]*data['spike_times'] + p[1]).astype('int32')
    return spike_times_npx, p


def make_plots(data, p):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    t_patch = np.arange(len(data['patch']))
    st = data['spike_times']

    # plot patch with spike times overlaid
    t_start = t_patch[st[0]] - 1000
    axs[0].plot(t_patch, data['patch'], label='patch')
    axs[0].plot(t_patch[st], data['patch'][st], '*', label='spike times')
    axs[0].set_xlim([t_start, t_start + 25000])  # plot 0.5 sec of data
    axs[0].legend()

    # plot sync pulses post-correction
    t_npx = np.arange(len(data['npx_sync']))
    t_patch_adj = t_patch*p[0] + p[1]
    axs[1].plot(t_npx, data['npx_sync'], label='npx_sync')
    axs[1].plot(t_patch_adj, data['patch_sync'],
                '--', label='patch_sync_aligned')
    axs[1].legend()
    axs[1].set_xlim([0, 100000])  # plot ~3 sec of data

def run(path, save_path, display_flag, row):
    print('Extracting spikes for cell: ' + row['Cell'])
    data = load_data(path, row['Cell'], row['chan_highest'])
    data['spike_times_np'], p = convert_patch_to_np_times(data)
    if display_flag:
        make_plots(data, p)
        plt.savefig(os.path.join(save_path, data['folder'], data['folder'] + '_gt_spikes_validation.png'))
    np.save(os.path.join(
        save_path, data['folder'], data['folder'] + '_gt_spikes'), data['spike_times_np'])
    print(os.path.join(
        save_path, data['folder']))

def run_all(path, save_path, display_flag):
    data_summary_df = load_data_summary(
        os.path.join(path, 'Data Summary.xlsx'))
    for index, row in data_summary_df.iterrows():
        try:
            run(path, save_path, display_flag, row)
        except:
            print(f"skipped {row['Cell']}")


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to extract ground truth spike times',
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('path', type=str,
                        help='Path to recordings (where cXX folders are contained)')
    parser.add_argument('--display', action='store_true',
                        help='Whether to generate plots')
    args = parser.parse_args()
    path = args.path
    save_path = args.path
    display_flag = args.display
    run_all(path, save_path, display_flag)
    

