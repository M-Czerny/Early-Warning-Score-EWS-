import numpy as np
import math
import json
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from scipy.signal import butter, filtfilt
from random import randint
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

def lowpass_filter(signal, ppg_fs):
    
    highcut=6.0
    order=4

    nyquist = 0.5 * ppg_fs
    normal_cutoff = highcut / nyquist

    b, a = butter(order, normal_cutoff, btype='low')

    return filtfilt(b, a, signal)

def highpass_filter(signal, ppg_fs):
    
    highcut=0.5
    order=4

    nyquist = 0.5 * ppg_fs
    normal_cutoff = highcut / nyquist

    b, a = butter(order, normal_cutoff, btype='high')

    return filtfilt(b, a, signal)


def get_SpO2_values(SpO2, update_time, spo2_fs):
    window = int(spo2_fs*update_time)
    SpO2_values = []
    for x in range(window, len(SpO2), window):
        SpO2_samp = np.mean(SpO2[x-window:x])
        SpO2_values.append(SpO2_samp) 

    return SpO2_values

def get_R_values(red, ir, sample_len, update_time, ppg_fs):
    window = int(ppg_fs*update_time)
    red_filt_hp = highpass_filter(red,ppg_fs)
    red_filt_lp = lowpass_filter(red,ppg_fs)
    ir_filt_hp = highpass_filter(ir,ppg_fs)
    ir_filt_lp = lowpass_filter(ir,ppg_fs)

    R_values = []
    for x in range(sample_len*ppg_fs, len(red), window):
        red_ppg_samp = red[x-sample_len*ppg_fs:x]
        red_filt_hp_samp = red_filt_hp[x-sample_len*ppg_fs:x]
        red_filt_lp_samp = red_filt_lp[x-sample_len*ppg_fs:x]
        ir_ppg_samp = ir[x-sample_len*ppg_fs:x]
        ir_filt_hp_samp = ir_filt_hp[x-sample_len*ppg_fs:x]
        ir_filt_lp_samp = ir_filt_lp[x-sample_len*ppg_fs:x]

        if ppg_check(red_ppg_samp, red_filt_hp_samp, ppg_fs) and ppg_check(ir_ppg_samp, ir_filt_hp_samp, ppg_fs):
            R = ratio(red_ppg_samp,red_filt_hp_samp,red_filt_lp_samp,ir_ppg_samp,ir_filt_hp_samp,ir_filt_lp_samp)
            R_values.append(R)
        else:
            R_values.append(math.nan)
        

    R_values = np.array(R_values).reshape(-1, 1)
    
    return R_values

def robust_ac(signal):
    p95 = np.percentile(signal, 95)
    p5  = np.percentile(signal, 5)
    return (p95 - p5)/2

def ratio(red_signal, red_filt_hp, red_filt_lp, ir_signal, ir_filt_hp, ir_filt_lp):
    
    '''
    plt.plot(red_signal,label='red')
    #plt.plot(bandpass_filter(red_signal,ppg_fs),label='red_bandpass')
    plt.plot(lowpass_filter(red_signal,ppg_fs),label='red_lowpass')
    plt.legend(loc='best')
    plt.show()
    '''

    red_AC = np.std(red_filt_hp)
    red_DC = np.mean(red_filt_lp)
    ir_AC = np.std(ir_filt_hp)
    ir_DC = np.mean(ir_filt_lp)

    #red_AC = np.max(red_filt_hp) - np.min(red_filt_hp)
    #red_DC = np.mean(red_filt_lp)
    #ir_AC = np.max(ir_filt_hp) - np.min(ir_filt_hp)
    #ir_DC = np.mean(ir_filt_lp)

    #red_AC = robust_ac(red_filt_hp)
    #red_DC = np.mean(red_signal)
    #ir_AC = robust_ac(ir_filt_hp)
    #ir_DC = np.mean(ir_signal)
    

    R = (red_AC/red_DC)/(ir_AC/ir_DC)

    return R

def acceleration(accx, accy, accz, update_time, ppg_fs):
    window = int(ppg_fs*update_time)
    ind = []
    acc = []
    for x in range(window, len(accx), window):
        a = np.sqrt(np.mean(accx[(x-window):x])**2 + np.mean(accy[(x-window):x])**2 + np.mean(accz[(x-window):x])**2)
        acc.append(a)
        #if a > 16060 or a <16045: # Session 1
        #if a > 16000 or a <12000: # Session 2
        if a < 16900 or a > 17035: # Session 3
            ind.append(int(x/window)-1)
            
        
    return ind, acc

def get_datetime(session):
    file_path = session

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("# {"):
                # Remove leading '# '
                json_str = line[2:].strip()
                header_dict = json.loads(json_str)
                break

    # The device MAC address is the main key
    device_key = list(header_dict.keys())[0]

    date_str = header_dict[device_key]["date"]
    time_str = header_dict[device_key]["time"]

    # Combine date + time
    datetime_str = f"{date_str} {time_str}"

    return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")

def get_testing_data(R_values, SpO2_values, window_len,train_percent):
    ind = randint(0, int(len(R_values)*train_percent)+1)
    
    test_R_values = R_values[ind:(ind+int(len(R_values)*(1-train_percent)))+1]
    test_SpO2_values = SpO2_values[ind:(ind+int(len(SpO2_values)*(1-train_percent)))+1]
    if ind >= window_len:
        train_R_values = np.concatenate((R_values[0:ind-window_len], R_values[ind+int(len(R_values)*(1-train_percent))+window_len:]), axis=0).reshape(-1, 1)
        train_SpO2_values = np.concatenate((SpO2_values[0:ind-window_len], SpO2_values[ind+int(len(SpO2_values)*(1-train_percent))+window_len:]), axis=0)
    else:
        train_R_values = R_values[ind+int(len(R_values)*(1-train_percent))+window_len:].reshape(-1, 1)
        train_SpO2_values = SpO2_values[ind+int(len(SpO2_values)*(1-train_percent))+window_len:]
    

    return train_R_values, train_SpO2_values, test_R_values, test_SpO2_values

def data_shuffle(R_values, SpO2_values, window_len):
    ind = shuffle(range(0, int(len(R_values)/window_len)))
    
    shuffled_R_values = np.empty([len(R_values)-len(R_values)%window_len,1])
    shuffled_SpO2_values = np.empty(len(SpO2_values)-len(SpO2_values)%window_len)

    for x in range(0,len(ind)):
        shuffled_R_values[window_len*x:window_len*x+window_len] = R_values[window_len*ind[x]:window_len*ind[x]+window_len]
        shuffled_SpO2_values[window_len*x:window_len*x+window_len] = SpO2_values[window_len*ind[x]:window_len*ind[x]+window_len]

    return shuffled_R_values, shuffled_SpO2_values

def calc_coefficients_SVR(R, SpO2):
    model = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="linear", C=10.0, epsilon=0.5))])
    model.fit(R, SpO2)
    svr = model.named_steps["svr"]
    scaler = model.named_steps["scaler"]

    w_scaled = svr.coef_[0][0]
    b_scaled = svr.intercept_[0]

    A = b_scaled - w_scaled * scaler.mean_[0] / scaler.scale_[0]
    B = -w_scaled / scaler.scale_[0]
    print('SVR: A =',A,' , B =',B)

    return A, B

def calc_coefficients_LR(R, SpO2):
    model = LinearRegression()
    model.fit(R, SpO2)

    A = model.intercept_
    B = -model.coef_[0]

    print('LR: A =',A,' , B =',B)

    return A, B

def calc_coefficients_HR(R, SpO2):
    model = HuberRegressor()
    model.fit(R, SpO2)

    A = model.intercept_
    B = -model.coef_[0]

    print('HR: A =',A,' , B =',B)

    return A, B

def calc_coefficients_RANSAC(R, SpO2):
    model = RANSACRegressor(LinearRegression())
    model.fit(R, SpO2)

    A = model.estimator_.intercept_
    B = -model.estimator_.coef_[0]

    print('RANSAC: A =',A,' , B =',B)

    return A, B