from pathlib import Path
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

def bandpass_filter(signal, ppg_fs):

    lowcut=0.5
    highcut=5
    order=5

    b, a = butter(order, [lowcut, highcut], fs=ppg_fs, btype='band')

    return filtfilt(b, a, signal)

def lowpass_filter(signal, ppg_fs):
    
    lowcut=0.5
    order=5

    b, a = butter(order, lowcut, fs=ppg_fs, btype='low')

    return filtfilt(b, a, signal)

def acc(ax, ay, az):
    return np.sqrt(ax**2 + ay**2 + az**2)

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

def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Z-score normalise, returning a float array."""
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

Subject = '5'
sample_len = 10
SpO2_fs= 100
ppg_fs= 50

base_dir = 'SpO2/Pilot Data/session3/subject'+Subject+'/'

files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
opensignals_file = files[2] if files else None


ppg_file = pd.read_csv(base_dir+files[0], delimiter="\t")
ir, green, red, blue, accx, accy, accz = ppg_file.get('IR'), ppg_file.get('Green'), ppg_file.get('Red'), ppg_file.get('Blue'), ppg_file.get('ACCx'), ppg_file.get('ACCy'), ppg_file.get('ACCz')
ppg_timestamp = int(files[0][:-4])

SpO2_file = np.loadtxt(base_dir+opensignals_file)
SpO2_values = SpO2_file[:,4]
SpO2_red = SpO2_file[:,3]
SpO2_ir = SpO2_file[:,2]


# Time Syncing
ppg_datetime = datetime.fromtimestamp(ppg_timestamp/1000)
SpO2_datetime = get_datetime(base_dir+opensignals_file)
time_diff = (ppg_datetime - SpO2_datetime).total_seconds() + sample_len # Adding sample_len for when R values can start to be calculated
print(time_diff)

if time_diff > 0:
    SpO2_values = SpO2_values[round(time_diff*SpO2_fs):]
    SpO2_red = SpO2_red[round(time_diff*SpO2_fs):]
    SpO2_ir = SpO2_ir[round(time_diff*SpO2_fs):]
else:
    ir, green, red, blue, accx, accy, accz = ir[round(-time_diff*ppg_fs):], green[round(-time_diff*ppg_fs):], red[round(-time_diff*ppg_fs):], blue[round(-time_diff*ppg_fs):], accx[round(-time_diff*ppg_fs):], accy[round(-time_diff*ppg_fs):], accz[round(-time_diff*ppg_fs):]

acc_mag = acc(accx, accy, accz)

Sp = []
Sp_red = []
Sp_ir = []
for x in range(2, len(SpO2_red), 2):
    Sp_samp = np.mean(SpO2_values[x-2:x])
    Sp.append(Sp_samp) 
    Sp_red_samp = np.mean(SpO2_red[x-2:x])
    Sp_red.append(Sp_red_samp) 
    Sp_ir_samp = np.mean(SpO2_ir[x-2:x])
    Sp_ir.append(Sp_ir_samp) 


red_bp = bandpass_filter(np.asarray(red),50)
ir_bp = bandpass_filter(np.asarray(ir),50)
red_lp = lowpass_filter(np.asarray(red),50)
ir_lp = lowpass_filter(np.asarray(ir),50)
Sp_red_bp = bandpass_filter(np.array(Sp_red),50)
Sp_ir_bp = bandpass_filter(np.array(Sp_ir),50)
Sp_red_lp = lowpass_filter(np.array(Sp_red),50)
Sp_ir_lp = lowpass_filter(np.array(Sp_ir),50)

start = 15000
stop = 15500


red_AC = np.std(red_bp[start:stop])
red_DC = np.mean(red_lp[start:stop])
ir_AC = np.std(ir_bp[start:stop])
ir_DC = np.mean(ir_lp[start:stop])
R = (red_AC/red_DC)/(ir_AC/ir_DC)
print("PPG Ratio: ", R) 
print("PPG Spo2: ", 110 - 25*R) 

ref_red_AC = np.std(Sp_red_bp[start:stop])
ref_red_DC = np.mean(Sp_red_lp[start:stop])
ref_ir_AC = np.std(Sp_ir_bp[start:stop])
ref_ir_DC = np.mean(Sp_ir_lp[start:stop])
ref_R = (ref_red_AC/ref_red_DC)/(ref_ir_AC/ref_ir_DC)
print("Ref Ratio: ", ref_R) 
print("Ref Spo2: ", 110 - 25*ref_R) 

plt.subplot(3, 1, 1)
#plt.plot(red[start:stop], label='red')
#plt.plot(ir[start:stop], label='ir')
plt.plot(red_bp[start:stop], label='red_bp')
plt.plot(ir_bp[start:stop], label='ir_bp')
plt.legend(loc='center right')
plt.subplot(3, 1, 2)
#plt.plot(Sp_red[start:stop], label='SpO2_red')
#plt.plot(Sp_ir[start:stop], label='SpO2_ir')
plt.plot(Sp_red_bp[start:stop], label='SpO2_red_bp')
plt.plot(Sp_ir_bp[start:stop], label='SpO2_ir_bp')
plt.legend(loc='center right')
plt.subplot(3, 1, 3)
plt.plot(Sp[start:stop], label='SpO2_ir')
plt.legend(loc='center right')
plt.show()


'''
plt.subplot(2, 1, 1)
plt.plot(blue, label='blue')
plt.plot(ir, label='ir')
plt.plot(green, label='green')
plt.plot(red, label='red')
plt.legend(loc='best')
plt.subplot(2, 1, 2)
plt.plot(Sp, label='spo2')
plt.legend(loc='best')
plt.show()
'''