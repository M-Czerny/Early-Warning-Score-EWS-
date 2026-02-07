import numpy as np
import sys
import math
import pandas as pd 
import datetime
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import wfdb
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class coefficients():

    A : float
    B : float
    sample_len = 5
    window_len = 10
    ppg_fs = 86
    spo2_fs = 2
    sao2_fs = 2

    def __init__(self,files):
        R_values = np.array([0])
        SpO2_values = np.array([0])

        for filename in files:
            print(filename)
            current_R_values, ppg_base_time = self.get_R_values(filename+'_ppg')
            current_SpO2_values, SpO2_base_time = self.get_SpO2_values(filename+'_2hz.csv')
            synced_R_values, synced_SpO2_values = self.time_sync(current_R_values, ppg_base_time, current_SpO2_values, SpO2_base_time)
            
            
            R_values = np.vstack((R_values,synced_R_values))
            SpO2_values = np.hstack((SpO2_values,synced_SpO2_values))
        
        shuffled_R_values, shuffled_SpO2_values = self.shuffle(R_values[1:-1], SpO2_values[1:-1])
        self.A, self.B = self.calc_coefficients(shuffled_R_values, shuffled_SpO2_values) 

    def get_R_values(self, filename):
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

        for x in range(self.sample_len, int(len(red_ppg_array)/self.ppg_fs)-1):
            red_ppg_samp = red_ppg_array[(x-self.sample_len)*self.ppg_fs:x*self.ppg_fs]
            ir_ppg_samp = ir_ppg_array[(x-self.sample_len)*self.ppg_fs:x*self.ppg_fs]

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

            if self.is_good_ppg(red_ppg_samp, self.bandpass_filter(red_ppg_samp)) and self.is_good_ppg(ir_ppg_samp, self.bandpass_filter(ir_ppg_samp)):
                R = self.ratio(red_ppg_samp,ir_ppg_samp)
                R_values.append(R)
            else:
                R_values.append(math.nan)

        R_values = np.array(R_values).reshape(-1, 1)
        
        ppg_base_time = (datetime.datetime.combine(datetime.datetime.now().date(),red_ppg[1].get('base_time')) + datetime.timedelta(0,self.sample_len)).time()

        return R_values, ppg_base_time

    def bandpass_filter(self, signal):

        lowcut=0.5
        highcut=6.0
        order=4

        nyquist = 0.5 * self.ppg_fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')

        return filtfilt(b, a, signal)
    
    def perfusion_index(self, ppg_raw, ppg_filt):
        ac = np.std(ppg_filt)
        dc = np.mean(ppg_raw)
        return ac / dc
    
    def heart_rate(self, ppg_filt):
        peaks,_ = find_peaks(ppg_filt, distance=self.ppg_fs*0.4)
        if len(peaks) < 2:
            return None
        rr = np.diff(peaks) / self.ppg_fs
        return 60 / np.mean(rr)
    
    def pulse_shape_consistency(self, ppg_filt):
        peaks,_ = find_peaks(ppg_filt, distance=self.ppg_fs*0.4)
        if len(peaks) < 3:
            return 0

        pulses = []
        win = int(0.6 * self.ppg_fs)

        for p in peaks[1:-1]:
            pulses.append(ppg_filt[p-win//2:p+win//2])

        ref = np.mean(pulses, axis=0)
        corrs = [np.corrcoef(ref, p)[0,1] for p in pulses]
        return np.mean(corrs)
    
    def spectral_sqi(self, ppg_filt):
        fft = np.abs(np.fft.rfft(ppg_filt))
        freqs = np.fft.rfftfreq(len(ppg_filt), 1/self.ppg_fs)

        hr_band = (freqs > 0.5) & (freqs < 4)
        return np.sum(fft[hr_band]) / np.sum(fft)

    def is_good_ppg(self, ppg_raw, ppg_filt):
        pi = self.perfusion_index(ppg_raw, ppg_filt)
        hr = self.heart_rate(ppg_filt)
        shape = self.pulse_shape_consistency(ppg_filt)
        spec = self.spectral_sqi(ppg_filt)

        if pi < 0.0005:
            return False
        if hr is None or hr < 40 or hr > 180:
            return False
        if shape < 0.6:
            return False
        if spec < 0.4:
            return False

        return True
    '''
    def nlms_filter(ppg, ref, mu=0.5, filter_len=8, eps=1e-6):
        N = len(ppg)
        w = np.zeros(filter_len)
        y = np.zeros(N)
        e = np.zeros(N)

        for n in range(filter_len, N):
            x = ref[n-filter_len:n][::-1]  
            y[n] = np.dot(w, x) 
            e[n] = ppg[n] - y[n] 

            norm = np.dot(x, x) + eps
            w = w + (mu / norm) * x * e[n]

        return e
    '''
    def ratio(self, red_signal, ir_signal):
        red_AC = np.std(self.bandpass_filter(red_signal))
        red_DC = np.mean(red_signal)
        ir_AC = np.std(self.bandpass_filter(ir_signal))
        ir_DC = np.mean(ir_signal)
        '''
        red_AC = np.max(self.bandpass_filter(red_signal)) - np.min(self.bandpass_filter(red_signal))
        red_DC = np.mean(red_signal)
        ir_AC = np.max(self.bandpass_filter(ir_signal)) - np.min(self.bandpass_filter(ir_signal))
        ir_DC = np.mean(ir_signal)
        '''

        R = (red_AC/red_DC)/(ir_AC/ir_DC)

        return R
    
    def get_SpO2_values(self, filename):

        #SpO2_values = np.ndarray.flatten(pd.read_csv(filename, usecols=['dev64_SpO2']).to_numpy())
        SpO2_values = np.ndarray.flatten(pd.read_csv(filename, usecols=['ScalcO2']).to_numpy())
        SpO2_values = SpO2_values[::self.spo2_fs] 

        SpO2_base_time = pd.read_csv(filename, usecols=['Timestamp'],nrows=1).to_numpy()[0,0][11:-1]

        return SpO2_values, SpO2_base_time
    
    def time_sync(self, R_values, ppg_base_time, SpO2_values, SpO2_base_time):
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
        
        synced_SpO2_values = np.delete(synced_SpO2_values,ind)
        synced_R_values = np.delete(synced_R_values,ind)
        synced_R_values = np.array(synced_R_values).reshape(-1, 1)

        return synced_R_values, synced_SpO2_values

    def shuffle(self, R_values, SpO2_values):
        ind = shuffle(range(0, int(len(R_values)/self.window_len)))
        
        shuffled_R_values = np.empty([len(R_values)-len(R_values)%self.window_len,1])
        shuffled_SpO2_values = np.empty(len(SpO2_values)-len(SpO2_values)%self.window_len)

        for x in range(0,len(ind)):
            shuffled_R_values[self.window_len*x:self.window_len*x+self.window_len] = R_values[self.window_len*ind[x]:self.window_len*ind[x]+self.window_len]
            shuffled_SpO2_values[self.window_len*x:self.window_len*x+self.window_len] = SpO2_values[self.window_len*ind[x]:self.window_len*ind[x]+self.window_len]

        return shuffled_R_values, shuffled_SpO2_values


    def calc_coefficients(self, R_values, SpO2_values):
        np.set_printoptions(threshold=sys.maxsize)
        print(R_values)
        print(SpO2_values)

        '''
        model = LinearRegression()
        model.fit(R_values, SpO2_values)

        A = model.intercept_
        B = -model.coef_[0]
        '''
        if len(R_values) > 0:
            model = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="linear", C=10.0, epsilon=0.5))])
            model.fit(R_values, SpO2_values)
            svr = model.named_steps["svr"]
            scaler = model.named_steps["scaler"]

            w_scaled = svr.coef_[0][0]
            b_scaled = svr.intercept_[0]

            A = b_scaled - w_scaled * scaler.mean_[0] / scaler.scale_[0]
            B = -w_scaled / scaler.scale_[0]

            print(A,B)
        else:
            A = 0
            B = 0

        return A, B
    
