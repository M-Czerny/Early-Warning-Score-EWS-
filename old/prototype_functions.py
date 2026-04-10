import numpy as np
import math
import json
import matplotlib.pyplot as plt
import pandas as pd
import os

from scipy.signal import find_peaks
from datetime import datetime
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from scipy.signal import butter, filtfilt, lfilter
from random import randint
from ppg_check import ppg_check


def bandpass_filter(signal, ppg_fs):

    lowcut=0.5
    highcut=5
    order=5

    b, a = butter(order, [lowcut, highcut], fs=ppg_fs, btype='band')

    return lfilter(b, a, signal)

def lowpass_filter(signal, ppg_fs):
    
    lowcut=0.5
    order=5

    b, a = butter(order, lowcut, fs=ppg_fs, btype='low')

    return lfilter(b, a, signal)

def highpass_filter(signal, ppg_fs):
    
    highcut=5
    order=5

    b, a = butter(order, highcut, fs=ppg_fs, btype='high')

    return lfilter(b, a, signal)


def get_SpO2_values(SpO2, update_time, spo2_fs):
    window = int(spo2_fs*update_time)
    SpO2_values = []
    for x in range(window, len(SpO2), window):
        SpO2_samp = np.mean(SpO2[x-window:x+window])
        SpO2_values.append(SpO2_samp) 

    return SpO2_values

def get_R_values(red, ir, sample_len, update_time, ppg_fs):
    window = int(ppg_fs*update_time)
    red_filt_bp = bandpass_filter(red,ppg_fs)
    red_filt_lp = lowpass_filter(red,ppg_fs)
    ir_filt_bp = bandpass_filter(ir,ppg_fs)
    ir_filt_lp = lowpass_filter(ir,ppg_fs)

    R_values = []
    for x in range(sample_len*ppg_fs, len(red), window):
        red_ppg_samp = red[x-sample_len*ppg_fs:x]
        red_filt_bp_samp = red_filt_bp[x-sample_len*ppg_fs:x]
        red_filt_lp_samp = red_filt_lp[x-sample_len*ppg_fs:x]
        ir_ppg_samp = ir[x-sample_len*ppg_fs:x]
        ir_filt_bp_samp = ir_filt_bp[x-sample_len*ppg_fs:x]
        ir_filt_lp_samp = ir_filt_lp[x-sample_len*ppg_fs:x]

        if ppg_check(red_ppg_samp, red_filt_bp_samp, ppg_fs) and ppg_check(ir_ppg_samp, ir_filt_bp_samp, ppg_fs):
            R = ratio(red_ppg_samp,red_filt_bp_samp,red_filt_lp_samp,ir_ppg_samp,ir_filt_bp_samp,ir_filt_lp_samp)
            R_values.append(R)
        else:
            R_values.append(math.nan)
        

    R_values = np.array(R_values).reshape(-1, 1)
    
    return R_values

def get_R_values_ver2(red, ir, accx, accy, accz, sample_len, update_time, ppg_fs):
    window = int(ppg_fs*update_time)

    R_values = []
    for x in range(sample_len*ppg_fs, len(red), window):
        red_ppg_samp = red[x-sample_len*ppg_fs:x]
        red_filt_bp_samp = bandpass_filter(red_ppg_samp,ppg_fs)
        red_filt_lp_samp = lowpass_filter(red_ppg_samp,ppg_fs)
        ir_ppg_samp = ir[x-sample_len*ppg_fs:x]
        ir_filt_bp_samp = bandpass_filter(ir_ppg_samp,ppg_fs)
        ir_filt_lp_samp = lowpass_filter(ir_ppg_samp,ppg_fs)

        red_bp_clean = lms_multi(red_filt_bp_samp, [accx, accy, accz, acc(accx, accy, accz)], mu=0.001, filter_order=5)
        red_lp_clean = lms_multi(red_filt_lp_samp, [accx, accy, accz, acc(accx, accy, accz)], mu=0.001, filter_order=5)
        ir_bp_clean = lms_multi(ir_filt_bp_samp, [accx, accy, accz, acc(accx, accy, accz)], mu=0.001, filter_order=5)
        ir_lp_clean = lms_multi(ir_filt_lp_samp, [accx, accy, accz, acc(accx, accy, accz)], mu=0.001, filter_order=5)

        if ppg_check(red_ppg_samp, red_lp_clean, ppg_fs) and ppg_check(ir_ppg_samp, ir_lp_clean, ppg_fs):
            R = ratio(red_ppg_samp,red_bp_clean,red_lp_clean,ir_ppg_samp,ir_bp_clean,ir_lp_clean)
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

def acc(ax, ay, az):
    return np.sqrt(ax**2 + ay**2 + az**2)

def lms_multi(ppg, acc_signals, mu=0.001, filter_order=5):
    """
    acc_signals: list or array of shape (n_signals, N)
                 e.g. [ax, ay, az, acc_mag]
    """
    acc_signals = np.array(acc_signals)
    n_signals, N = acc_signals.shape

    # Initialize weights
    w = np.zeros((n_signals, filter_order))
    
    # Output
    y = np.zeros(N)  # estimated noise
    e = np.zeros(N)  # cleaned signal

    # Pad signals
    padded = np.pad(acc_signals, ((0,0),(filter_order-1,0)))

    for i in range(N):
        x_all = []
        for j in range(n_signals):
            x = padded[j, i:i+filter_order][::-1]
            x_all.append(x)

        x_all = np.array(x_all)

        # Estimated noise
        y[i] = np.sum(w * x_all)

        # Error (clean signal)
        e[i] = ppg[i] - y[i]

        # Update weights
        w += 2 * mu * e[i] * x_all

    return e

def get_Data(Session,Subject,Episode,ppg_fs,SpO2_fs,sample_len,update_time):
    if Subject == 2:
        folder = 'SpO2/Pilot Data/session{}/subject{}/episode{}/'.format(Session,Subject,Episode)
        h = 17500
    else:
        folder = 'SpO2/Pilot Data/session{}/subject{}/'.format(Session,Subject)
        h = 18000

    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    opensignals_file = files[2] if files else None


    ppg_file = pd.read_csv(folder+files[0], delimiter="\t")
    ir, green, red, blue, accx, accy, accz = ppg_file.get('IR'), ppg_file.get('Green'), ppg_file.get('Red'), ppg_file.get('Blue'), ppg_file.get('ACCx'), ppg_file.get('ACCy'), ppg_file.get('ACCz')
    ppg_timestamp = int(files[0][:-4])

    SpO2_file = np.loadtxt(folder+opensignals_file)
    SpO2_values = SpO2_file[:,4]

    # Time Syncing
    ppg_datetime = datetime.fromtimestamp(ppg_timestamp/1000)
    SpO2_datetime = get_datetime(folder+opensignals_file)
    time_diff = (ppg_datetime - SpO2_datetime).total_seconds() + sample_len # Adding sample_len for when R values can start to be calculated
    print(time_diff)

    if time_diff > 0:
        SpO2_values = SpO2_values[round(time_diff*SpO2_fs):]
    else:
        ir, green, red, blue, accx, accy, accz = ir[round(-time_diff*ppg_fs):], green[round(-time_diff*ppg_fs):], red[round(-time_diff*ppg_fs):], blue[round(-time_diff*ppg_fs):], accx[round(-time_diff*ppg_fs):], accy[round(-time_diff*ppg_fs):], accz[round(-time_diff*ppg_fs):]

    acc_mag = acc(accx, accy, accz)
    
    Sp = []
    for x in range(2, len(SpO2_values), 2):
        Sp_samp = np.mean(SpO2_values[x-2:x])
        Sp.append(Sp_samp) 
    
    # Red, IR, Blue, Green PPG signals 
    # Acceleration in each axis 
    plt.subplot(2, 1, 1)
    plt.plot(blue, label='blue')
    plt.plot(ir, label='ir')
    plt.plot(green, label='green')
    plt.plot(red, label='red')
    plt.plot(acc_mag, label='acc_mag')
    plt.legend(loc='center right')
    plt.subplot(2, 1, 2)
    plt.plot(Sp, label='SpO2')
    plt.legend(loc='center right')
    plt.show()
    

    #w = 1
    #wl=None
    #peak_idx, _ = find_peaks(x = acc_mag, width = w, wlen=wl, height=h,prominence=None,distance=None)
    
    #for n in range(len(peak_idx)-1,0,-1):
    #    ir, green, red, blue, accx, accy, accz, acc_mag = np.hstack((ir[:peak_idx[n]],ir[peak_idx[n]+ppg_fs*sample_len:])), np.hstack((green[:peak_idx[n]],green[peak_idx[n]+ppg_fs*sample_len:])), np.hstack((red[:peak_idx[n]],red[peak_idx[n]+ppg_fs*sample_len:])), np.hstack((blue[:peak_idx[n]],blue[peak_idx[n]+ppg_fs*sample_len:])), np.hstack((accx[:peak_idx[n]],accx[peak_idx[n]+ppg_fs*sample_len:])), np.hstack((accy[:peak_idx[n]],accy[peak_idx[n]+ppg_fs*sample_len:])), np.hstack((accz[:peak_idx[n]],accz[peak_idx[n]+ppg_fs*sample_len:])), np.hstack((acc_mag[:peak_idx[n]],acc_mag[peak_idx[n]+ppg_fs*sample_len:]))
    #    SpO2_values = np.hstack((SpO2_values[:peak_idx[n]*int(SpO2_fs/ppg_fs)],SpO2_values[peak_idx[n]*int(SpO2_fs/ppg_fs)+SpO2_fs*sample_len:]))
        
    
    

    '''
    # Red, IR, Blue, Green PPG signals 
    # Acceleration in each axis 
    print(peak_idx)
    plt.subplot(2, 1, 1)
    plt.plot(blue, label='blue')
    plt.plot(ir, label='ir')
    plt.plot(green, label='green')
    plt.plot(red, label='red')
    plt.plot(acc_mag, label='acc_mag')
    plt.legend(loc='center right')
    plt.subplot(2, 1, 2)
    plt.plot(SpO2_values, label='SpO2')
    plt.legend(loc='center right')
    plt.show()
    '''

    # Getting SpO2 and R Values 
    SpO2 = get_SpO2_values(SpO2_values,update_time,SpO2_fs)
    R = get_R_values(red,ir,sample_len,update_time,ppg_fs)
    RG = get_R_values(red,green,sample_len,update_time,ppg_fs)
    RB = get_R_values(red,blue,sample_len,update_time,ppg_fs)
    IR = get_R_values(ir,red,sample_len,update_time,ppg_fs)
    IG = get_R_values(ir,green,sample_len,update_time,ppg_fs)
    IB = get_R_values(ir,blue,sample_len,update_time,ppg_fs)
    GR = get_R_values(green,red,sample_len,update_time,ppg_fs)
    GI = get_R_values(green,ir,sample_len,update_time,ppg_fs)
    GB = get_R_values(green,blue,sample_len,update_time,ppg_fs)
    BR = get_R_values(blue,red,sample_len,update_time,ppg_fs)
    BI = get_R_values(blue,ir,sample_len,update_time,ppg_fs)
    BG = get_R_values(blue,green,sample_len,update_time,ppg_fs)
    
    #R = get_R_values_ver2(red,ir,accx,accy,accz,sample_len,update_time,ppg_fs)
    length = min(len(R), len(SpO2))
    SpO2 = SpO2[:length]
    R = R[:length]
    RG = RG[:length]
    RB = RB[:length]
    IR = IR[:length]
    IG = IG[:length]
    IB = IB[:length]
    GR = GR[:length]
    GI = GI[:length]
    GB = GB[:length]
    BR = BR[:length]
    BI = BI[:length]
    BG = BG[:length]


    '''
    # Acceleration filtering
    # Removing data when acceleration squared sum reaches threshold
    ac_ind, acc = acceleration(accx, accy, accz, update_time, ppg_fs)
    
    new_acc = np.delete(acc,ac_ind)
    ac_ind = [x for x in ac_ind if x < length]

    R = np.delete(R,ac_ind)
    SpO2 = np.delete(SpO2,ac_ind)


    plt.plot(new_acc, label='acc')
    plt.legend(loc='best')
    plt.show()
    '''



    # Remove SpO2 data dropping under certain value
    min_spo2 = 90
    R = np.array([R[x] for x in range(0,len(SpO2)) if SpO2[x] >= min_spo2]).flatten()
    RG = np.array([RG[x] for x in range(0,len(SpO2)) if SpO2[x] >= min_spo2]).flatten()
    RB = np.array([RB[x] for x in range(0,len(SpO2)) if SpO2[x] >= min_spo2]).flatten()
    IR = np.array([IR[x] for x in range(0,len(SpO2)) if SpO2[x] >= min_spo2]).flatten()
    IG = np.array([IG[x] for x in range(0,len(SpO2)) if SpO2[x] >= min_spo2]).flatten()
    IB = np.array([IB[x] for x in range(0,len(SpO2)) if SpO2[x] >= min_spo2]).flatten()
    GR = np.array([GR[x] for x in range(0,len(SpO2)) if SpO2[x] >= min_spo2]).flatten()
    GI = np.array([GI[x] for x in range(0,len(SpO2)) if SpO2[x] >= min_spo2]).flatten()
    GB = np.array([GB[x] for x in range(0,len(SpO2)) if SpO2[x] >= min_spo2]).flatten()
    BR = np.array([BR[x] for x in range(0,len(SpO2)) if SpO2[x] >= min_spo2]).flatten()
    BI = np.array([BI[x] for x in range(0,len(SpO2)) if SpO2[x] >= min_spo2]).flatten()
    BG = np.array([BG[x] for x in range(0,len(SpO2)) if SpO2[x] >= min_spo2]).flatten()
    SpO2 = [x for x in SpO2 if x >= min_spo2]

    # Remove Nan
    SpO2 = [SpO2[x] for x in range(0,len(R)) if not math.isnan(R[x])]
    R = np.array([x for x in R if not math.isnan(x)]).reshape(-1, 1)
    RG = np.array([x for x in RG if not math.isnan(x)]).reshape(-1, 1)
    RB = np.array([x for x in RB if not math.isnan(x)]).reshape(-1, 1)
    IR = np.array([x for x in IR if not math.isnan(x)]).reshape(-1, 1)
    IG = np.array([x for x in IG if not math.isnan(x)]).reshape(-1, 1)
    IB = np.array([x for x in IB if not math.isnan(x)]).reshape(-1, 1)
    GR = np.array([x for x in GR if not math.isnan(x)]).reshape(-1, 1)
    GI = np.array([x for x in GI if not math.isnan(x)]).reshape(-1, 1)
    GB = np.array([x for x in GB if not math.isnan(x)]).reshape(-1, 1)
    BR = np.array([x for x in BR if not math.isnan(x)]).reshape(-1, 1)
    BI = np.array([x for x in BI if not math.isnan(x)]).reshape(-1, 1)
    BG = np.array([x for x in BG if not math.isnan(x)]).reshape(-1, 1)

    #plt.plot(SpO2, label='SpO2')
    #plt.plot(98.3-0.255*R, label='R')
    #plt.legend(loc='best')
    #plt.show()

    return R, SpO2, RG, RB, IR, IG, IB, GR, GI, GB, BR, BI, BG

def k_fold_cross_validation(k, R, SpO2, window_len, train_percent):
    A_list = [] 
    B_list = []
    for x in range(0,k):
        train_R, train_SpO2, test_R, test_SpO2 = get_testing_data(R, SpO2, window_len, train_percent)
        windowed_train_R, windowed_train_SpO2 = data_shuffle(train_R, train_SpO2, window_len)

        A_LR,B_LR = calc_coefficients_LR(windowed_train_R, windowed_train_SpO2)
        A_list.append(A_LR)
        B_list.append(B_LR)

    A = np.mean(A_list)
    B = np.mean(B_list)
    print(A)
    print(B)

    plt.plot(A - B*R, label='LinearR')
    plt.plot(SpO2, label='SpO2')

    plt.legend(loc='best')
    plt.show()
    return