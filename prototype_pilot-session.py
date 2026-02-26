import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
from prototype_functions import get_SpO2_values, get_R_values, acceleration, get_datetime, data_shuffle, calc_coefficients_SVR, calc_coefficients_LR, calc_coefficients_HR, calc_coefficients_RANSAC


ppg_fs = 50
SpO2_fs = 200

sample_len = 6 # How many seconds of past ppg signal used to calculate Ratio
update_time = 1 # Interval when new Ratio is calculated
window_len = 10 # Window size of signals used to shuffle for regression

Session = '2'



# Fetching PPG and SpO2 Data
folder = 'SpO2/Pilot Data/session'+Session+'/'
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
opensignals_file = files[1] if files else None

ppg_file = pd.read_csv(folder+'/prototype.txt', delimiter="\t")
ppg_timestamp, ir, green, red, blue, accx, accy, accz = ppg_file.get('SessionStartEpochMs'), ppg_file.get('IR'), ppg_file.get('Green'), ppg_file.get('Red'), ppg_file.get('Blue'), ppg_file.get('ACCx'), ppg_file.get('ACCy'), ppg_file.get('ACCz')

SpO2_file = np.loadtxt(folder+opensignals_file)
SpO2_values = SpO2_file[:,4]




# Time Syncing
ppg_datetime = datetime.fromtimestamp(ppg_timestamp[0]/1000)
SpO2_datetime = get_datetime(folder+opensignals_file)
time_diff = (ppg_datetime - SpO2_datetime).total_seconds() + sample_len # Adding sample_len for when R values can start to be calculated

if time_diff > 0:
    SpO2_values = SpO2_values[round(time_diff*SpO2_fs):]
else:
    ir, green, red, blue, accx, accy, accz = ir[round(-time_diff*ppg_fs):], green[round(-time_diff*ppg_fs):], red[round(-time_diff*ppg_fs):], blue[round(-time_diff*ppg_fs):], accx[round(-time_diff*ppg_fs):], accy[round(-time_diff*ppg_fs):], accz[round(-time_diff*ppg_fs):]




# Getting SpO2 and R Values 
SpO2 = get_SpO2_values(SpO2_values,update_time,SpO2_fs)
#R = get_R_values(red,ir,sample_len,update_time,ppg_fs)
R = get_R_values(blue,green,sample_len,update_time,ppg_fs)
length = min(len(R), len(SpO2))
SpO2 = SpO2[:length]
R = R[:length]




# Acceleration filtering
# Removing data when acceleration squared sum reaches threshold
ac_ind, acc = acceleration(accx, accy, accz, update_time, ppg_fs)
new_acc = np.delete(acc,ac_ind)
ac_ind = [x for x in ac_ind if x < length]

R = np.delete(R,ac_ind)
SpO2 = np.delete(SpO2,ac_ind)




# Remove SpO2 data dropping under 90
R = [R[x] for x in range(0,len(SpO2)) if SpO2[x] >= 90]
SpO2 = [x for x in SpO2 if x >= 90]

# Remove Nan
SpO2 = [SpO2[x] for x in range(0,len(R)) if not math.isnan(R[x])]
R = np.array([x for x in R if not math.isnan(x)]).reshape(-1, 1)



# Window the PPG & SpO2 Data
# Separating to training(80%) and testing(20%) data
windowed_R, windowed_SpO2 = data_shuffle(R, SpO2, window_len)
train_R = windowed_R[:int(len(windowed_R)*0.80)]
train_SpO2 = windowed_SpO2[:int(len(windowed_SpO2)*0.80)]
test_R = windowed_R[int(len(windowed_R)*0.80):]
test_SpO2 = windowed_SpO2[int(len(windowed_SpO2)*0.80):]



# Calculate Coefficents
A_SVR,B_SVR = calc_coefficients_SVR(train_R, train_SpO2)
A_LR,B_LR = calc_coefficients_LR(train_R, train_SpO2)
A_HR,B_HR = calc_coefficients_HR(train_R, train_SpO2)
A_RANSAC,B_RANSAC = calc_coefficients_RANSAC(train_R, train_SpO2)



## Plotting

# Test results from different Regressors and SpO2
plt.plot(A_SVR - B_SVR*test_R, label='SVR')
plt.plot(A_LR - B_LR*test_R, label='LR')
plt.plot(A_HR - B_HR*test_R, label='HR')
plt.plot(A_RANSAC - B_RANSAC*test_R, label='RANSAC')
plt.plot(test_SpO2, label='SpO2')
plt.legend(loc='best')
plt.show()

# Errors different Regressors to SpO2 and Errors
test_R = test_R.reshape(len(test_R))
plt.plot(A_SVR - B_SVR*test_R - test_SpO2, label='SVR_error')
plt.plot(A_LR - B_LR*test_R - test_SpO2, label='LR_error')
plt.plot(A_HR - B_HR*test_R - test_SpO2, label='HR_error')
plt.plot(A_RANSAC - B_RANSAC*test_R - test_SpO2, label='RANSAC_error')
plt.legend(loc='best')
print('SVR-MAPE Error: ',np.mean(abs(A_SVR - B_SVR*test_R - test_SpO2)/test_SpO2)*100,'%')
print('LR-MAPE Error: ',np.mean(abs(A_LR - B_LR*test_R - test_SpO2)/test_SpO2)*100,'%')
print('HR-MAPE Error: ',np.mean(abs(A_HR - B_HR*test_R - test_SpO2)/test_SpO2)*100,'%')
print('RANSAC-MAPE Error: ',np.mean(abs(A_RANSAC - B_RANSAC*test_R - test_SpO2)/test_SpO2)*100,'%')
plt.show()
'''
# Red, IR, Blue, Green PPG signals 
plt.plot(blue, label='blue')
plt.plot(ir, label='ir')
plt.plot(green, label='green')
plt.plot(red, label='red')
plt.legend(loc='best')
plt.show()

# Squared Sum of acceleration 
plt.plot(acc, label='acc')
plt.legend(loc='best')
plt.show()


# Acceleration in each axis 
plt.plot(accx, label='accx')
plt.plot(accy, label='accy')
plt.plot(accz, label='accz')
plt.legend(loc='best')
plt.show()
'''