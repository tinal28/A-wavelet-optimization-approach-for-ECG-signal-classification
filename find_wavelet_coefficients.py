import numpy as np
import pywt
import matplotlib.pyplot as plt
import find_filter_coefficients as find_fc

def wavelet_coefficients(signal, dec_lo, dec_hi, level=1):
    # Time-reverse low_dec for low_rec
    rec_lo = dec_lo[::-1]
    # Time-reverse hi_dec and change signs for every other element to get hi_rec
    rec_hi = [(-1)**i * dec_hi[::-1][i] for i in range(len(dec_hi))]
    
    filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]
    custom_wavelet = pywt.Wavelet(name="custom_wavelet", filter_bank=filter_bank)
    
    # Perform the Discrete Wavelet Transform
    coeffs = pywt.wavedec(signal, custom_wavelet, level=level, mode='symmetric')
    
    # coeffs[0]: Approximation coefficients at level 3 (the coarsest representation).
    # coeffs[1]: Detail coefficients at level 3.
    # coeffs[2]: Detail coefficients at level 2.
    # coeffs[3]: Detail coefficients at level 1.
    
    ceoffs_concat = np.concatenate(coeffs)
    # Return the approximation (low-pass) and detail (high-pass) coefficients , concatenated coefficients
    return coeffs, ceoffs_concat

   

# return updated wavelet coefficients for new positions     
def update_wavelet_coefficients(X_train, lowpass_filter_bank, highpass_filter_bank, levels, coeff_type):

    S = lowpass_filter_bank.shape[0]
    
    if coeff_type == 'a':
        var = 0
    elif coeff_type == 'd':
        var = 1

    # Prepare the dataset for the fitness function evaluation
    X_train_combined =[]
    
    #training datawith morphological features
    for j in range(S):
        features = []
        for i in range(X_train.shape[0]):
            # 1-3 temporal features
            # 3-303 features ECG beats  
            wc, _ = wavelet_coefficients(X_train.iloc[i,3:], lowpass_filter_bank[j], highpass_filter_bank[j], levels)
            temporal_features= X_train.iloc[i,1:3]
            features.append(np.concatenate((wc[var],temporal_features)))
        X_train_combined.append(features)

    X_train_combined =np.array(X_train_combined)
    return X_train_combined