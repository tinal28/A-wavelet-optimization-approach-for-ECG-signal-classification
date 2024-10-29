from wfdb import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    Prepare the dataset for the classification task.
    """
    fs = 360  # Sampling rate of the MIT-BIH records
    directory = "C:\\Users\\tinal\\Desktop\\sem 7\\Bio Signal Processing\\paper implementation wawelet\\dataset\\mit-bih-arrhythmia-database-1.0.0\\"  # Path to the database

    X = []  # To store features
    y = []  # To store labels

    for rec_no in records:
        try:
            # Load ECG record and annotations, using channel 1 for record 114
            if rec_no == 114:  
                ecg_record, ecg_annotations = read_data(directory, rec_no, ch=1)
            else:
                ecg_record, ecg_annotations = read_data(directory, rec_no)

        except FileNotFoundError:
            print(f"Record {rec_no} not found")
            continue

        # Extract the samples and symbols
        samples = ecg_annotations.sample
        ecg_signal = ecg_record.p_signal[:, 0]  # Only take the first channel (lead 1)
        symbols = ecg_annotations.symbol

        beat_symbols = []
        R_peaks = []

        # Filter for beat symbols only
        for i in range(len(samples)):
            if symbols[i] not in ["[", "]", "x", "(", ")", "p", "t", "u", "`", "'", "^", "|", "~", "+", "s", "T",
                                      "*", "D", "=", "@"]:
                beat_symbols.append(symbols[i])
                R_peaks.append(samples[i])
                
        # Convert to numpy arrays
        beat_symbols = np.array(beat_symbols)
        R_peaks = np.array(R_peaks)

        # Extract features for each valid beat
        for i, r_peak in enumerate(R_peaks):
            x = np.zeros(303)  # Initialize with 303 features per beat

            # Skip first 10 and last 5 beats
            if i < 10 or i > len(R_peaks) - 5:
                continue

            # Calculate left and right indices based on RR interval
            left_index = r_peak - (R_peaks[i] - R_peaks[i-1]) // 2
            right_index = r_peak + (R_peaks[i+1] - R_peaks[i]) // 2
            ecg_beat = ecg_signal[left_index:right_index]
            
            # Normalize and assign features
            ecg_beat = period_normaliztion(ecg_beat, 300)
            x[0] = None  # Placeholder for QRS duration
            x[1] = (R_peaks[i] - R_peaks[i-1]) / fs  # Previous RR interval
            x[2] = mean_rr_interval(R_peaks[i-10:i]) / fs  # Mean RR interval of last 10 beats
            x[3:303] = ecg_beat  # ECG beat data
            
            # Add data to X and y if the beat symbol is in the specified classes
            if beat_symbols[i] in classes:
                X.append(x)
                y.append(beat_symbols[i])

    # Create the final DataFrame from X and y after all records are processed
    dataset = pd.DataFrame(X)
    dataset['label'] = y

    return dataset

# Example usage
req_records = [100, 102, 104, 105, 106, 107, 118, 119, 200, 201, 202, 203, 205, 
                    208, 209, 212, 213, 214, 215, 217]
req_classes = ["N", "L", "R", "A", "V", "/"]

data = load_mit_datset(req_records, req_classes)
