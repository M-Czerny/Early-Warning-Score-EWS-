import numpy as np
import datetime
import math
import pandas as pd 
import wfdb
from sklearn.utils import shuffle
from scipy.signal import butter, filtfilt
from ppg_check import ppg_check

def bandpass_filter(signal, ppg_fs):

    lowcut=0.5
    highcut=6.0
    order=4

    nyquist = 0.5 * ppg_fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, signal)

def get_R_values(filename, sample_len, ppg_fs):
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
        print(self.is_good_ppg(red_ppg_samp, self.bandpass_filter(red_ppg_samp)))
        print(self.is_good_ppg(ir_ppg_samp, self.bandpass_filter(ir_ppg_samp)))
        print(self.ratio(red_ppg_samp,ir_ppg_samp))
        plt.plot(red_ppg_samp,label='Red')
        plt.plot(ir_ppg_samp,label='IR')
        plt.xlabel('seconds')
        plt.legend(loc='best')
        plt.show()
        '''

        if ppg_check(red_ppg_samp, bandpass_filter(red_ppg_samp, ppg_fs),ppg_fs) and ppg_check(ir_ppg_samp, bandpass_filter(ir_ppg_samp, ppg_fs),ppg_fs):
            R = ratio(red_ppg_samp,ir_ppg_samp,ppg_fs)
            R_values.append(R)
        else:
            R_values.append(math.nan)

    R_values = np.array(R_values).reshape(-1, 1)
    
    ppg_base_time = (datetime.datetime.combine(datetime.datetime.now().date(),red_ppg[1].get('base_time')) + datetime.timedelta(0,sample_len)).time()

    return R_values, ppg_base_time

def ratio(red_signal, ir_signal, ppg_fs):
        
    red_AC = np.std(bandpass_filter(red_signal,ppg_fs))
    red_DC = np.mean(red_signal)
    ir_AC = np.std(bandpass_filter(ir_signal,ppg_fs))
    ir_DC = np.mean(ir_signal)
    '''
    red_AC = np.max(self.bandpass_filter(red_signal)) - np.min(self.bandpass_filter(red_signal))
    red_DC = np.mean(red_signal)
    ir_AC = np.max(self.bandpass_filter(ir_signal)) - np.min(self.bandpass_filter(ir_signal))
    ir_DC = np.mean(ir_signal)
    '''

    R = (red_AC/red_DC)/(ir_AC/ir_DC)

    return R

def get_SpO2_values(filename, sample_len, spo2_fs):

    #SpO2_values = np.ndarray.flatten(pd.read_csv(filename, usecols=['dev64_SpO2']).to_numpy())
    SpO2_values = np.ndarray.flatten(pd.read_csv(filename, usecols=['ScalcO2']).to_numpy())
    SpO2_values = SpO2_values[::spo2_fs] 

    SpO2_base_time = pd.read_csv(filename, usecols=['Timestamp'],nrows=1).to_numpy()[0,0][11:-1]

    return SpO2_values, SpO2_base_time

def time_sync( R_values, ppg_base_time, SpO2_values, SpO2_base_time):
    SpO2_base_time = datetime.time(int(SpO2_base_time[0:2]),int(SpO2_base_time[3:5]),int(SpO2_base_time[6:8]),int(SpO2_base_time[9:11]+'0000'))

    ppg_datetime = datetime.datetime.combine(datetime.datetime.now().date(),ppg_base_time)
    SpO2_datetime = datetime.datetime.combine(datetime.datetime.now().date(),SpO2_base_time)

    if ppg_datetime > SpO2_datetime:
        synced_SpO2_values = SpO2_values[(ppg_datetime-SpO2_datetime).seconds:]
        synced_R_values = R_values
    else:
        synced_SpO2_values = SpO2_values
        synced_R_values = R_values[(SpO2_datetime-ppg_datetime).seconds:]

    if len(synced_R_values) > len(synced_SpO2_values):
        synced_R_values = synced_R_values[0:len(synced_SpO2_values)]
    else:
        synced_SpO2_values = synced_SpO2_values[0:len(synced_R_values)]

    #print(np.concatenate((synced_R_values, synced_SpO2_values.reshape(len(synced_SpO2_values),1)),axis=1))

    ind=[]
    for x in range(len(synced_SpO2_values)-1, -1, -1):
        if math.isnan(synced_SpO2_values[x]) or math.isnan(synced_R_values[x][0]):
            #synced_SpO2_values = np.delete(synced_SpO2_values,x)
            #synced_R_values = np.delete(synced_R_values,x)
            ind.append(x)
    
    synced_SpO2_values = np.delete(synced_SpO2_values,ind)
    synced_R_values = np.delete(synced_R_values,ind)
    synced_R_values = np.array(synced_R_values).reshape(-1, 1)

    return synced_R_values, synced_SpO2_values

def data_shuffle(R_values, SpO2_values, window_len):
    ind = shuffle(range(0, int(len(R_values)/window_len)))
    
    shuffled_R_values = np.empty([len(R_values)-len(R_values)%window_len,1])
    shuffled_SpO2_values = np.empty(len(SpO2_values)-len(SpO2_values)%window_len)

    for x in range(0,len(ind)):
        shuffled_R_values[window_len*x:window_len*x+window_len] = R_values[window_len*ind[x]:window_len*ind[x]+window_len]
        shuffled_SpO2_values[window_len*x:window_len*x+window_len] = SpO2_values[window_len*ind[x]:window_len*ind[x]+window_len]

    return shuffled_R_values, shuffled_SpO2_values
