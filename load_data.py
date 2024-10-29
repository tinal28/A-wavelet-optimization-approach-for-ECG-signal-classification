from wfdb import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import find_filter_coefficients as find_fc
import find_wavelet_coefficients as fwc

def read_data(dir, record_number, ch=0):
    """
    Read ECG signals and annotations

    Args:
        dir: Path to the dataset
        record_number: Record Number or Name
        ch: channel number

    Returns:
        ecg_record, ecg_ann : ECG signal and it's annotations.

    """

    record_name = dir + str(record_number)
    ecg_record = rdrecord(record_name, channels=[ch])
    ecg_ann = rdann(record_name, extension='atr')
    return ecg_record, ecg_ann

# TF????
def period_normaliztion(y, num_samples):
    n = num_samples
    n_ = len(y)
    x = np.zeros(n)
    for j in range(n):
        rj = (j-1)*(n_-1)/(n-1) 
        j_ = int(rj)
        x[j] = y[j_] + (y[j_+1]- y[j_])*(rj-j_)
    return x
    
def mean_rr_interval(r_peaks):
    
    # find the RR intervals
    rr_intervals = np.diff(r_peaks)
    
    return np.mean(rr_intervals)

def load_mit_datset(records, classes):

    """
    Prepare the dataset for the classification task
    
    """
    fs = 360  # sampling rate of the MIT-BIH records

    directory =  "C:\\Users\\tinal\\Desktop\\sem 7\\Bio Signal Processing\\paper implementation wawelet\\dataset\\mit-bih-arrhythmia-database-1.0.0\\" # path to the database
    dataset = pd.DataFrame()
    X = []
    y = []
    
    for rec_no in records:
        try:
            if rec_no == 114:  # MLII is channel 1 not 0
                ecg_record, ecg_annotations = read_data(directory, rec_no, ch=1)
            else:
                ecg_record, ecg_annotations = read_data(directory, rec_no)

        except FileNotFoundError:
            print(f"Record {rec_no} not found")
            continue

        samples = ecg_annotations.sample
        ecg_signal = ecg_record.p_signal[:, 0]  # only take the first channel ( lead 1)
        symbols = ecg_annotations.symbol

        beat_symbols = []
        R_peaks = []

        # Remove non-beat annotations
        for i in range(len(samples)):
            if symbols[i] not in ["[", "]", "x", "(", ")", "p", "t", "u", "`", "'", "^", "|", "~", "+", "s", "T",
                                      "*", "D", "=", "@"]:
                beat_symbols.append(symbols[i])
                R_peaks.append(samples[i])
                
        # Convert to numpy arrays
        beat_symbols = np.array(beat_symbols)
        R_peaks = np.array(R_peaks)
        
        for i, r_peak in enumerate(R_peaks):
            
            x = np.zeros(303) # 303 features per beat
            
            # neglect first 10 ecg beats and last 5 ecg beats
            if i < 10  or i > len(R_peaks) - 5:
                continue

            ## TF?
            else:
                left_index = r_peak - (R_peaks[i]-R_peaks[i-1])//2
                right_index = r_peak + (R_peaks[i+1]-R_peaks[i])//2
                ecg_beat = ecg_signal[left_index:right_index]
                
                ecg_beat = period_normaliztion(ecg_beat, 300)
                x[0] = None # QRS duration
                x[1] = (R_peaks[i] - R_peaks[i-1])/fs # Previous RR interval
                x[2] = (mean_rr_interval(R_peaks[i-10:i]))/fs # mean RR interval of the last 10 beats
                x[3:303] = ecg_beat
                if beat_symbols[i] in classes:
                    X.append(x)
                    y.append(beat_symbols[i])
        
        dataset = pd.DataFrame(X)
        dataset['label'] = y
    
    return dataset
    
# return updated wavelet coefficients for new positions     
def update_wavelet_coefficients(X_train,positions):

    #  iFind FIR low-pass filter coefficients for each particle
    S,d= positions.shape[0],positions.shape[1]
    # Initializ an array of size [S,2*d]
    lowpass_filter_bank = np.zeros((S,2*d))
    highpass_filter_bank = np.zeros((S,2*d))

    # find the low-pass filter coefficients for each particle   
    for particle in range(S):
        x = find_fc.lowpass_filter(positions[particle])
        lowpass_filter_bank[particle] = x

    # find high-pass filter coefficients for each particle from low-pass filter coefficients
    for particle in range(S):
        x = find_fc.high_pass_filter(lowpass_filter_bank[particle])
        highpass_filter_bank[particle] = x 

    # Perform DWT with custom filters and decomposition level 2
    X_train_combined =[]
    #training datawith morphological features
    for j in range(S):
        features = []
        for i in range(X_train.shape[0]):
            _,wavelet_coefficients = fwc.wavelet_coefficients(X_train.iloc[i,3:], lowpass_filter_bank[j], highpass_filter_bank[j], 2)
            temporal_features= X_train.iloc[i,1:3]
            features.append(np.concatenate((wavelet_coefficients,temporal_features)))
        X_train_combined.append(features)

    X_train_combined =np.array(X_train_combined)
    # print("train set size",X_train_combined.shape)

    return X_train_combined
    
###
#load dataset
# 1-3 temporal features
# 3-303 features ECG beats  