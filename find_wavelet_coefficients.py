import numpy as np
import pywt
import matplotlib.pyplot as plt

def wavelet_coefficients(signal, dec_lo, dec_hi, level=1):
    # Time-reverse low_dec for low_rec
    rec_lo = dec_lo[::-1]
    # Time-reverse hi_dec and change signs for every other element to get hi_rec
    rec_hi = [(-1)**i * dec_hi[::-1][i] for i in range(len(dec_hi))]
    
    filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]
    custom_wavelet = pywt.Wavelet(name="custom_wavelet", filter_bank=filter_bank)
    
    # Perform the Discrete Wavelet Transform
    coeffs = pywt.wavedec(signal, custom_wavelet, level=level, mode='symmetric')
    ceoffs_concat = np.concatenate(coeffs)
    # Return the approximation (low-pass) and detail (high-pass) coefficients , concatenated coefficients
    return coeffs, ceoffs_concat

# coeffs[0]: Approximation coefficients at level 3 (the coarsest representation).
# coeffs[1]: Detail coefficients at level 3.
# coeffs[2]: Detail coefficients at level 2.
# coeffs[3]: Detail coefficients at level 1.