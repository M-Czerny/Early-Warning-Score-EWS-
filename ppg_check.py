import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.signal import welch



def ppg_check(ppg_window, ppg_filt, ppg_fs):

    features = {
        "PI": perfusion_index(ppg_window, ppg_filt),
        "Skewness": ppg_skewness(ppg_filt),
        "Kurtosis": ppg_kurtosis(ppg_filt),
        "SNR": ppg_snr(ppg_filt, ppg_fs),
        "Entropy": spectral_entropy(ppg_filt, ppg_fs),
        "RelativePower": relative_power(ppg_filt, ppg_fs),
        "HeartRate": heart_rate(ppg_filt, ppg_fs),
        "HeartRate": heart_rate(ppg_filt, ppg_fs),
        "PulseShapeConsistency": pulse_shape_consistency(ppg_filt, ppg_fs),
        "SQI": spectral_sqi(ppg_filt, ppg_fs),
    }
    
    pi = features['PI']
    sk = features['Skewness']
    ku = features['Kurtosis']
    ppg_SNR = features['SNR']
    en = features['Entropy']
    rp = features['RelativePower']
    hr = features['HeartRate']
    shape = features['PulseShapeConsistency']
    spec_SQI = features['SQI']

    
    if pi < 0.01:
        
        return False
    if sk < 0 or sk > 0.4:
        return False
    
    if ku < -1.5 or ku > 8:
        return False
    
    if ppg_SNR < 1.5:
        return False
    
    if rp < 0.5:
        return False
    
    if en > 5:
        return False
    
    #if shape < 0.2:
    #    return False
    
    if spec_SQI < 0.4:
        return False
    
    if hr is None or hr < 40 or hr > 180:
        return False
    

    return True
    
def perfusion_index(ppg_raw, ppg_filt):
    ac = np.std(ppg_filt)
    dc = np.mean(ppg_raw)
    return ac / dc if dc != 0 else np.nan

def ppg_skewness(ppg_filt):
    return skew(ppg_filt)

def ppg_kurtosis(ppg_filt):
    return kurtosis(ppg_filt, fisher=True)

def ppg_snr(ppg_filt, ppg_fs):
    f, Pxx = welch(ppg_filt, fs=ppg_fs, nperseg=256)

    hr_band = (f >= 0.5) & (f <= 4)
    signal_power = np.sum(Pxx[hr_band])
    noise_power = np.sum(Pxx[~hr_band])

    return signal_power / noise_power if noise_power != 0 else np.nan

def spectral_entropy(ppg_filt, ppg_fs):
    f, Pxx = welch(ppg_filt, fs=ppg_fs)
    Pxx_norm = Pxx / np.sum(Pxx)
    entropy = -np.sum(Pxx_norm * np.log2(Pxx_norm + 1e-12))
    return entropy

def relative_power(ppg_filt, ppg_fs):
    f, Pxx = welch(ppg_filt, fs=ppg_fs)

    hr_band = (f >= 0.5) & (f <= 4)
    return np.sum(Pxx[hr_band]) / np.sum(Pxx)

def heart_rate(ppg_filt, ppg_fs):
    peaks,_ = find_peaks(ppg_filt, distance=ppg_fs*0.4)
    if len(peaks) < 2:
        return None
    rr = np.diff(peaks) / ppg_fs
    return 60 / np.mean(rr)

def pulse_shape_consistency(ppg_filt, ppg_fs):
    peaks,_ = find_peaks(ppg_filt, distance=ppg_fs*0.4)
    if len(peaks) < 3:
        return 0

    pulses = []
    win = int(0.6 * ppg_fs)

    for p in peaks[1:-1]:
        pulses.append(ppg_filt[p-win//2:p+win//2])

    ref = np.mean(pulses, axis=0)
    corrs = [np.corrcoef(ref, p)[0,1] for p in pulses]
    return np.mean(corrs)

def spectral_sqi(ppg_filt, ppg_fs):
    fft = np.abs(np.fft.rfft(ppg_filt))
    freqs = np.fft.rfftfreq(len(ppg_filt), 1/ppg_fs)

    hr_band = (freqs > 0.5) & (freqs < 4)
    return np.sum(fft[hr_band]) / np.sum(fft)