import wave
import pandas as pd
import numpy as np
import python_speech_features as ps
import os
import pickle

def read_file(filename):
    """
    Fetch audio signal data, time elapsed and frame rate for a .wav file
    """
    file = wave.open(filename, 'r')    
    params = file.getparams()
    # Fetch parameters
    nchannels, sampwidth, framerate, nframes = params[:4]
    # Read and return a string of bytes
    str_data = file.readframes(nframes)
    wavedata = np.fromstring(str_data, dtype = np.short)
    time = np.arange(0, nframes) * (1.0 / framerate)
    file.close()
    return wavedata, time, framerate

def get_fixed_length(data, time, framerate, 
                     start = 0.5, end = 3.5, pad_value = 0.0):
    """
    Generate data with fixed start and end
    Input:
    data:      audio signal data returned by read_file function
    time:      time elapsed array returned by read_file function
    framerate: frame rate returned by read_file function
    start:     start for trimming audio signal (second)
    end:       end for trimming audio signal (second)
    pad_value: numerical value used for padding when data is shorter than `end`

    Output:
    data_new:  filtered bytes data [start, end]
    """
    # Filter data to the specified range
    data_new = data[np.where((time > start) & (time <= end))]
    # Add padding when needed
    nframes = (end - start) * framerate
    if len(data_new) <= nframes:
        data_new = np.pad(data_new, (0, int(nframes - len(data_new))), "constant", constant_values=(pad_value))
    return data_new

def extract_feature(path, start = 0.5, end = 3.5, pad_value = 0.0):
    """
    Extract log Mel-filterbank energy, delta and delta-delta of audio signal
    Input:      
    path:       directory path
    start:     start for trimming audio signal (second)
    end:       end for trimming audio signal (second)
    pad_value: numerical value used for padding when data is shorter than `end`

    Output:
    log_fbank_raw:   Log Mel-filterbank energy
    delta_raw:       delta
    delta_delta_raw: delta-delta
    """
    
    # Create placeholders for features
    log_fbank_raw = []
    delta_raw = []
    delta_delta_raw = []

    for subdir, dirs, files in os.walk(path):
        for file_name in files:
            if ".wav" in file_name:
                # Read bytes, time elapsed and frame rate
                data, time, framerate = read_file(os.path.join(subdir, file_name))
                # Create fixed length data
                data = get_fixed_length(data, time, framerate, start, end)
                # Compute log Mel-filterbank energy
                log_fbank = ps.logfbank(data, framerate, nfilt = 40, nfft = 1200)
                log_fbank_raw.append(log_fbank)
                # Compute delta
                delta = ps.delta(log_fbank, 2)
                delta_raw.append(delta)
                # Compute delta-delta
                delta_delta = ps.delta(delta, 2)
                delta_delta_raw.append(delta_delta)
    return log_fbank_raw, delta_raw, delta_delta_raw

def normalize_feature(feature, train_ind, val_ind):
    """
    Normalize features using training data
    """
    # Compute mean and standard deviation using training data only
    feature_mean = np.mean(feature[train_ind], axis = 0)
    feature_std = np.std(feature[train_ind], axis = 0)
    # Normalize all observations
    feature = (feature - feature_mean) / feature_std
    return feature
