import spikeinterface as si
import spikeinterface.sorters as ss
import probeinterface as pi

import os
import glob
import time
from tqdm import tqdm

def main():
    dataset_ids = os.listdir('/hdd/kampff/raw/Recordings')
    for dataset_id in tqdm(dataset_ids):
        
        recording_filename = glob.glob(f'/hdd/kampff/raw/Recordings/{dataset_id}/{dataset_id}_npx_raw*.bin')
        recording = si.read_binary(file_paths=recording_filename, sampling_frequency=30e3, num_chan=384, dtype='int16')
        probe_group = pi.read_probeinterface("/home/kyu/repos/spike-sorting-hackathon/datasets/examples/NP1_standard_config.json")
        recording = recording.set_probegroup(probe_group)
        print(f'loaded recording {dataset_id}.')

        recording = si.preprocessing.bandpass_filter(recording=recording, freq_min=300, freq_max=5000)
        
        print(f'running KS3 on recording {dataset_id}...')
        t1 = time.time()
        sorting = ss.run_kilosort3(recording=recording,
                                   output_folder=f"/hdd/kampff/raw/Recordings/{dataset_id}/kilosort3",
                                   docker_image=True)
        t2 = time.time()
        print(f'finished running KS3 on recording {dataset_id} in {t2-t1}s.')

        recording_di = si.load_extractor('di')
                
        print(f'running KS3 on DI recording {dataset_id}...')
        t1 = time.time()
        sorting_di = ss.run_kilosort3(recording=recording_di,
                                      output_folder=f"/hdd/kampff/raw/Recordings/{dataset_id}/kilosort3_di",
                                      docker_image=True)
        t2 = time.time()
        print(f'finished running KS3 on DI recording {dataset_id} in {t2-t1}s.')
        
if __name__ == '__main__':
    main()
