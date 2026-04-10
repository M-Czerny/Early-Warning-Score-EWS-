import numpy as np
import pandas as pd
import os
from pathlib import Path
import wfdb
import csv
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from ppg_check import ppg_check
import math
import datetime
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



#SpO2_file = h5py.File('SpO2/opensignals_842e140cd8ef_2026-02-06_17-07-03.h5','r')


#SpO2_file = pd.read_csv('SpO2/opensignals_842e140cd8ef_2026-02-06_17-07-03.csv')

#ppg_file = open('SpO2/prototype.txt')



def bandpass_filter(signal, ppg_fs):

    lowcut=0.5
    highcut=6.0
    order=4

    nyquist = 0.5 * ppg_fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, signal)

def get_R_values(red, ir, sample_len, ppg_fs):
    R_values = []

    for x in range(sample_len, int(len(red)/ppg_fs)-1):
        red_ppg_samp = red[(x-sample_len)*ppg_fs:x*ppg_fs]
        ir_ppg_samp = ir[(x-sample_len)*ppg_fs:x*ppg_fs]

        R = ratio(red_ppg_samp,ir_ppg_samp,ppg_fs)
        R_values.append(R)
        #if ppg_check(red_ppg_samp, bandpass_filter(red_ppg_samp, ppg_fs),ppg_fs) and ppg_check(ir_ppg_samp, bandpass_filter(ir_ppg_samp, ppg_fs),ppg_fs):
        #    R = ratio(red_ppg_samp,ir_ppg_samp,ppg_fs)
        #    R_values.append(R)
        #else:
        #    R_values.append(math.nan)

    R_values = np.array(R_values).reshape(-1, 1)
    
    return R_values

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

def get_SpO2_values(SpO2, sample_len, spo2_fs):
    SpO2_values = []
    for x in range(0, int(len(SpO2)/spo2_fs)):
        SpO2_samp = np.mean(SpO2[x*spo2_fs:(x+1)*spo2_fs])
        SpO2_values.append(SpO2_samp) 

    return SpO2_values

def acc(accx, accy, accz, ppg_fs):
    ind = []
    for x in range(0, int(len(accx)/ppg_fs)-1):
        if np.mean(accx[x*ppg_fs:(x+1)*ppg_fs])**2 + np.mean(accy[x*ppg_fs:(x+1)*ppg_fs])**2 + np.mean(accz[x*ppg_fs:(x+1)*ppg_fs])**2 < 2.6e8:
            ind.append(x)
    return ind


ppg_file = pd.read_csv('SpO2/prototype.txt', delimiter="\t")
ppg_data = np.array(ppg_file[1:])
SpO2_values = np.loadtxt('SpO2/opensignals_842e140cd8ef_2026-02-06_17-07-03.txt')[:,4]


ppg_fs = 50
SpO_fs = 200
ppg_datetime = ppg_data[0,0]
SpO2_datetime = 1770752300000
time_diff = (ppg_datetime - SpO2_datetime)/1e8

SpO2 = get_SpO2_values(SpO2_values,10,SpO_fs)

# TIME DIFF NOT DONE PROPERLY
#SpO2 = SpO2[int(-time_diff):]

ir, green, red, blue, accx, accy, accz = ppg_data[:,2], ppg_data[:,3], ppg_data[:,4], ppg_data[:,5], ppg_data[:,6], ppg_data[:,7], ppg_data[:,8]
R = get_R_values(red,ir,10,ppg_fs)
ac_ind = acc(accx, accy, accz, ppg_fs)

R = np.delete(R,ac_ind[:-10])
SpO2 = np.delete(SpO2,ac_ind[:-10])

ind=[]
for x in range(len(SpO2)-1, -1, -1):
    if math.isnan(SpO2[x]) or math.isnan(R[x]):
        ind.append(x)

R = np.delete(R,ind)
R = np.array(R).reshape(-1, 1)
SpO2 = np.delete(SpO2,ind)

print(R)
print(SpO2)


model = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="linear", C=10.0, epsilon=0.5))])
model.fit(R, SpO2)
svr = model.named_steps["svr"]
scaler = model.named_steps["scaler"]

w_scaled = svr.coef_[0][0]
b_scaled = svr.intercept_[0]

A = b_scaled - w_scaled * scaler.mean_[0] / scaler.scale_[0]
B = -w_scaled / scaler.scale_[0]
print(A,B)

plt.plot(R)
plt.plot(SpO2)
plt.show()


print(SpO2)
plt.plot(blue, label='blue')
plt.plot(ir, label='ir')
plt.plot(green, label='green')
plt.plot(red, label='red')
plt.plot(accx, label='accx')
plt.plot(accy, label='accy')
plt.plot(accz, label='accz')
plt.legend(loc='best')
plt.show()
'''

ac = []
for x in range(0, int(len(accx)/ppg_fs)-1):
    ac.append(np.mean(accx[x*ppg_fs:(x+1)*ppg_fs])**2 + np.mean(accy[x*ppg_fs:(x+1)*ppg_fs])**2 + np.mean(accz[x*ppg_fs:(x+1)*ppg_fs])**2)

plt.plot(ac)
plt.show()
'''