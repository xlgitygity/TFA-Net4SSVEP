from scipy.io import loadmat
import numpy as np
import os
import time
import pywt


def slice_and_cwt(subject_order, freq_min, freq_max, length, overlap, Sampling_rate=250):

    samples = length * Sampling_rate  # Total number of samples per window
    step = overlap * Sampling_rate  # Step size between windows

    # Compute the number of windows we need to process
    total_samples = 1250  # Total length of data
    num_windows = (total_samples - samples) // step + 1  # Calculate how many windows we need
    trial_num = num_windows * 40 *6

    # Initialize the data storage for wavelet transform
    data_wt = np.zeros([trial_num, int((freq_max - freq_min) * 5 + 1), samples, 8], 'float32')

    # Load raw data
    data_raw = loadmat(f'/path/to/benchmark/S{subject_order}.mat')['data']  # [64, 1500, 40, 6]
    data_raw = np.concatenate([data_raw[53:58, :, :, :], data_raw[60:63, :, :, :]], axis=0)  # [8, 1500, 40, 6]

    # Prepare reshaped data
    data_reshape = np.zeros([240, 8, 1500])  # Reshaped data size
    data = np.zeros([trial_num, 8, samples])  # Final data for CWT

    # Reshape the raw data (for each block and target)
    for block in range(6):
        for target in range(40):
            data_reshape[block * 40 + target, :, :] = data_raw[:, :, target, block]

    # Slice data based on `samples` and `step`, creating sliding windows
    for i in range(num_windows):
        for samples_idx in range(samples):
            # Slice data based on `samples` and `step`
            data[i * 240: i * 240 + 240, :, samples_idx] = data_reshape[:, :, samples_idx + i * step + 125]  # skip the initial 0.5-s before stimulus onset

    # Wavelet transform parameters
    totalscal = 625
    fs = Sampling_rate
    wavelet = 'morl'
    wcf = pywt.central_frequency(wavelet=wavelet)
    cparam = 2 * wcf * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)

    # Perform the CWT for each trial and channel
    start = time.time()
    for trial in range(trial_num):
        for chans in range(8):
            cwtmatr, frequencies = pywt.cwt(data[trial, chans, :], scales, wavelet, 1.0 / fs, axis=2)
            data_wt[trial, :, :, chans] = cwtmatr[totalscal - int(freq_max * 5): totalscal - int(freq_min * 5) + 1, :]

    print(f"Time taken for CWT: {time.time() - start} seconds")

    # Save the processed data
    output_dir = '/path/to/your/file'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_windows):
        for block in range(6):
            for target in range(40):
                data_info = data_wt[i * 240 + 40 * block + target, :, :, :]
                filename = f"{subject_order:02d}{i}{block}_{target}.npy"
                np.save(os.path.join(output_dir, filename), data_info)

    print(f"Subject {subject_order} processed.")



# Example usage
subject_order = 1
freq_min = 8
freq_max = 31.8
slice_and_cwt(subject_order, freq_min, freq_max, 1, 0.5)