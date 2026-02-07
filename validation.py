import numpy as np
import math
import pandas as pd 
import datetime
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import wfdb

sample_len = 5
ppg_fs = 86
spo2_fs = 2
sao2_fs = 2

def validate(A, B, filename):

    print(filename)
    current_R_values, ppg_base_time = get_R_values(filename+'_ppg')
    current_SpO2_values, SpO2_base_time = get_SpO2_values(filename+'_2hz.csv')
    synced_R_values, synced_SpO2_values = time_sync(current_R_values, ppg_base_time, current_SpO2_values, SpO2_base_time)

    #print(synced_SpO2_values)
    plt.plot(A - B*synced_R_values,label='A - B * R')
    plt.plot(synced_SpO2_values,label='SaO2')
    plt.xlabel('seconds')
    plt.legend(loc='best')
    plt.show()

def get_R_values(filename):
    if wfdb.rdsamp(filename)[1].get('sig_name').count('Red Signal') == 1:
        red_ppg = wfdb.rdsamp(filename,channel_names=['Red Signal'])
        ir_ppg = wfdb.rdsamp(filename,channel_names=['IR Signal'])
    elif wfdb.rdsamp(filename)[1].get('sig_name').count('RED') == 1:
        red_ppg = wfdb.rdsamp(filename,channel_names=['RED'])
        ir_ppg = wfdb.rdsamp(filename,channel_names=['IR'])
    else:
        print(wfdb.rdsamp(filename)[1].get('sig_name'))

    red_ppg_array = np.ndarray.flatten(red_ppg[0])
    ir_ppg_array = np.ndarray.flatten(ir_ppg[0])

    R_values = []

    for x in range(sample_len, int(len(red_ppg_array)/ppg_fs)-1):
        red_ppg_samp = red_ppg_array[(x-sample_len)*ppg_fs:x*ppg_fs]
        ir_ppg_samp = ir_ppg_array[(x-sample_len)*ppg_fs:x*ppg_fs]
        
        '''
        R = ratio(red_ppg_samp,ir_ppg_samp)
        R_values.append(R)

        '''
        if is_good_ppg(red_ppg_samp, bandpass_filter(red_ppg_samp)) and is_good_ppg(ir_ppg_samp, bandpass_filter(ir_ppg_samp)):
            R = ratio(red_ppg_samp,ir_ppg_samp)
            R_values.append(R)
        else:
                R_values.append(math.nan)

    R_values = np.array(R_values).reshape(-1, 1)
    
    ppg_base_time = (datetime.datetime.combine(datetime.datetime.now().date(),red_ppg[1].get('base_time')) + datetime.timedelta(0,sample_len)).time()

    return R_values, ppg_base_time

def bandpass_filter(signal):

    lowcut=0.5
    highcut=6.0
    order=4

    nyquist = 0.5 * ppg_fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, signal)

def perfusion_index(ppg_raw, ppg_filt):
    ac = np.std(ppg_filt)
    dc = np.mean(ppg_raw)
    return ac / dc

def heart_rate(ppg_filt):
    peaks,_ = find_peaks(ppg_filt, distance=ppg_fs*0.4)
    if len(peaks) < 2:
        return None
    rr = np.diff(peaks) / ppg_fs
    return 60 / np.mean(rr)

def pulse_shape_consistency(ppg_filt):
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

def spectral_sqi(ppg_filt):
    fft = np.abs(np.fft.rfft(ppg_filt))
    freqs = np.fft.rfftfreq(len(ppg_filt), 1/ppg_fs)

    hr_band = (freqs > 0.5) & (freqs < 4)
    return np.sum(fft[hr_band]) / np.sum(fft)

def is_good_ppg(ppg_raw, ppg_filt):
    pi = perfusion_index(ppg_raw, ppg_filt)
    hr = heart_rate(ppg_filt)
    shape = pulse_shape_consistency(ppg_filt)
    spec = spectral_sqi(ppg_filt)

    if pi < 0.0005:
        return False
    if hr is None or hr < 40 or hr > 180:
        return False
    if shape < 0.6:
        return False
    if spec < 0.4:
        return False

    return True

def ratio(red_signal, ir_signal):
    red_AC = np.std(bandpass_filter(red_signal))
    red_DC = np.mean(red_signal)
    ir_AC = np.std(bandpass_filter(ir_signal))
    ir_DC = np.mean(ir_signal)
    '''
    red_AC = np.max(bandpass_filter(red_signal)) - np.min(bandpass_filter(red_signal))
    red_DC = np.mean(red_signal)
    ir_AC = np.max(bandpass_filter(ir_signal)) - np.min(bandpass_filter(ir_signal))
    ir_DC = np.mean(ir_signal)
    '''

    R = (red_AC/red_DC)/(ir_AC/ir_DC)

    return R

def get_SpO2_values(filename):

    #SpO2_values = np.ndarray.flatten(pd.read_csv(filename, usecols=['dev64_SpO2']).to_numpy())
    SpO2_values = np.ndarray.flatten(pd.read_csv(filename, usecols=['ScalcO2']).to_numpy())
    SpO2_values = SpO2_values[::spo2_fs] 

    SpO2_base_time = pd.read_csv(filename, usecols=['Timestamp'],nrows=1).to_numpy()[0,0][11:-1]

    return SpO2_values, SpO2_base_time

def time_sync(R_values, ppg_base_time, SpO2_values, SpO2_base_time):
    SpO2_base_time = datetime.time(int(SpO2_base_time[0:2]),int(SpO2_base_time[3:5]),int(SpO2_base_time[6:8]),int(SpO2_base_time[9:11]+'0000'))

    ppg_datetime = datetime.datetime.combine(datetime.datetime.now().date(),ppg_base_time)
    SpO2_datetime = datetime.datetime.combine(datetime.datetime.now().date(),SpO2_base_time)

    if ppg_datetime > SpO2_datetime:
        synced_SpO2_values = SpO2_values[(ppg_datetime-SpO2_datetime).seconds:-1]
        synced_R_values = R_values
    else:
        synced_SpO2_values = SpO2_values
        synced_R_values = R_values[(SpO2_datetime-ppg_datetime).seconds:-1]

    if len(synced_R_values) > len(synced_SpO2_values):
        synced_R_values = synced_R_values[0:len(synced_SpO2_values)]
    else:
        synced_SpO2_values = synced_SpO2_values[0:len(synced_R_values)]

    ind=[]
    for x in range(len(synced_SpO2_values)-1, -1, -1):
        if math.isnan(synced_SpO2_values[x]) or math.isnan(synced_R_values[x][0]):
            #synced_SpO2_values = np.delete(synced_SpO2_values,x)
            #synced_R_values = np.delete(synced_R_values,x)
            ind.append(x)
    
    
    
    
    #synced_SpO2_values = np.delete(synced_SpO2_values,ind)
    #synced_R_values = np.delete(synced_R_values,ind)
    #synced_R_values = np.array(synced_R_values).reshape(-1, 1)

    synced_R_values = np.ndarray.flatten(synced_R_values)
    #print(synced_SpO2_values)

    synced_SpO2_values[ind] = np.nan
    synced_R_values[ind] = np.nan

    return synced_R_values, synced_SpO2_values