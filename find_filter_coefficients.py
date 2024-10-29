import numpy as np

def lowpass_filter(alpha):
    # Constructs an array h(1 ... N) of lowpass orthonormal FIR filter coefficients
    # for any even N >= 2.

    ###############################################
    # starts from c0 and s0, 
    
    N = 2 * len(alpha)
    h = np.zeros(N)
    lo = N // 2 - 1
    hi = lo + 1
    
    h[lo] = np.cos(alpha[0])
    h[hi] = np.sin(alpha[0])
    
    nstages = N // 2
    
    for stage in range(1, nstages):
        c = np.cos(alpha[stage])
        s = np.sin(alpha[stage])
        
        h[lo-1] = c * h[lo]
        h[lo]   = s * h[lo]
        h[hi+1] = c * h[hi]
        h[hi]   = -s * h[hi]
        
        nbutterflies = stage - 1
        butterflybase = lo + 1
        
        for butterfly in range(nbutterflies):
            hlo = h[butterflybase]
            hhi = h[butterflybase + 1]
            h[butterflybase]     = c * hhi - s * hlo
            h[butterflybase + 1] = s * hhi + c * hlo
        
        lo -= 1
        hi += 1
    
    return h

def high_pass_filter(h):
    N = len(h) // 2  # Assuming h has even length and corresponds to low-pass filter coefficients
    g = np.zeros(len(h))
    
    for i in range(len(h)):
        g[i] = (-1) ** (i + 1) * h[2*N - 1 - i]
    
    return g