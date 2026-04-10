import numpy as np
import matplotlib.pyplot as plt
from prototype_functions import k_fold_cross_validation, get_testing_data, data_shuffle, calc_coefficients_SVR, calc_coefficients_LR, calc_coefficients_HR, calc_coefficients_RANSAC, lms_multi, get_Data


ppg_fs = 50
SpO2_fs = 100 # since 3rd session

sample_len = 10 # How many seconds of past ppg signal used to calculate Ratio
update_time = 1 # Interval when new Ratio is calculated
window_len = 10 # Window size of signals used to shuffle for regression
train_percent = 0.80 # Percentage of data used to train coefficients

Session = 3
Subject = 6
Episode = 2


# Fetching PPG and SpO2 Data
R, SpO2, RG, RB, IR, IG, IB, GR, GI, GB, BR, BI, BG = get_Data(Session,Subject,Episode,ppg_fs,SpO2_fs,sample_len,update_time)
'''
R_1, SpO2_1 = get_Data(3,1,1,ppg_fs,SpO2_fs,sample_len,update_time)
R_2, SpO2_2 = get_Data(3,2,1,ppg_fs,SpO2_fs,sample_len,update_time)
R_3, SpO2_3 = get_Data(3,2,2,ppg_fs,SpO2_fs,sample_len,update_time)
R_4, SpO2_4 = get_Data(3,3,1,ppg_fs,SpO2_fs,sample_len,update_time)
train_R = np.concatenate([R_1.flatten(),R_4.flatten()]).reshape(-1, 1)
train_SpO2 = np.hstack((SpO2_1,SpO2_4))
test_R = np.concatenate([R_2.flatten(),R_3.flatten()]).reshape(-1, 1)
test_SpO2 = np.hstack((SpO2_2,SpO2_3))
'''
# Window the PPG & SpO2 Data
# Separating to training(80%) and testing(20%) data without data leakage
#train_R, train_SpO2, test_R, test_SpO2 = get_testing_data(R, SpO2, window_len, train_percent)
#windowed_train_R, windowed_train_SpO2 = data_shuffle(train_R, train_SpO2, window_len)

# K-Fold Cross Validation
k_fold_cross_validation(20, GB, SpO2, window_len, train_percent)


'''
train_ratio = 0.8
R_train = R[:int(len(R)*train_ratio)]
RG_train = RG[:int(len(R)*train_ratio)]
RB_train = RB[:int(len(R)*train_ratio)]
IR_train = IR[:int(len(R)*train_ratio)]
IG_train = IG[:int(len(R)*train_ratio)]
IB_train = IB[:int(len(R)*train_ratio)]
GR_train = GR[:int(len(R)*train_ratio)]
GI_train = GI[:int(len(R)*train_ratio)]
GB_train = GB[:int(len(R)*train_ratio)]
BR_train = BR[:int(len(R)*train_ratio)]
BI_train = BI[:int(len(R)*train_ratio)]
BG_train = BG[:int(len(R)*train_ratio)]
SpO2_train = SpO2[:int(len(R)*train_ratio)]
R_test = R[int(len(R)*train_ratio)+1:]
RG_test = RG[int(len(R)*train_ratio)+1:]
RB_test = RB[int(len(R)*train_ratio)+1:]
IR_test = IR[int(len(R)*train_ratio)+1:]
IG_test = IG[int(len(R)*train_ratio)+1:]
IB_test = IB[int(len(R)*train_ratio)+1:]
GR_test = GR[int(len(R)*train_ratio)+1:]
GI_test = GI[int(len(R)*train_ratio)+1:]
GB_test = GB[int(len(R)*train_ratio)+1:]
BR_test = BR[int(len(R)*train_ratio)+1:]
BI_test = BI[int(len(R)*train_ratio)+1:]
BG_test = BG[int(len(R)*train_ratio)+1:]
SpO2_test = SpO2[int(len(R)*train_ratio)+1:]

A_R,B_R = calc_coefficients_LR(R_train, SpO2_train)
A_RG,B_RG = calc_coefficients_LR(RG_train, SpO2_train)
A_RB,B_RB = calc_coefficients_LR(RB_train, SpO2_train)
A_IR,B_IR = calc_coefficients_LR(IR_train, SpO2_train)
A_IG,B_IG = calc_coefficients_LR(IG_train, SpO2_train)
A_IB,B_IB = calc_coefficients_LR(IB_train, SpO2_train)
A_GR,B_GR = calc_coefficients_LR(GR_train, SpO2_train)
A_GI,B_GI = calc_coefficients_LR(GI_train, SpO2_train)
A_GB,B_GB = calc_coefficients_LR(GB_train, SpO2_train)
A_BR,B_BR = calc_coefficients_LR(BR_train, SpO2_train)
A_BI,B_BI = calc_coefficients_LR(BI_train, SpO2_train)
A_BG,B_BG = calc_coefficients_LR(BG_train, SpO2_train)



ax = plt.subplot(6, 2, 1)
plt.plot(A_R - B_R*R_test, label='Red/IR', color='r')
plt.plot(SpO2_test, label='SpO2', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
ax = plt.subplot(6, 2, 2)
plt.plot(A_RG - B_RG*RG_test, label='Red/Green', color='r')
plt.plot(SpO2_test, label='SpO2', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
ax = plt.subplot(6, 2, 3)
plt.plot(A_RB - B_RB*RB_test, label='Red/Blue', color='r')
plt.plot(SpO2_test, label='SpO2', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
ax = plt.subplot(6, 2, 4)
plt.plot(A_IR - B_IR*IR_test, label='IR/Red', color='orange')
plt.plot(SpO2_test, label='SpO2', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
ax = plt.subplot(6, 2, 5)
plt.plot(A_IG - B_IG*IG_test, label='IR/Green', color='orange')
plt.plot(SpO2_test, label='SpO2', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
ax = plt.subplot(6, 2, 6)
plt.plot(A_IB - B_IB*IB_test, label='IR/Blue', color='orange')
plt.plot(SpO2_test, label='SpO2', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
ax = plt.subplot(6, 2, 7)
plt.plot(A_GR - B_GR*GR_test, label='Green/Red', color='limegreen')
plt.plot(SpO2_test, label='SpO2', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
ax = plt.subplot(6, 2, 8)
plt.plot(A_GI - B_GI*GI_test, label='Green/IR', color='limegreen')
plt.plot(SpO2_test, label='SpO2', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
ax = plt.subplot(6, 2, 9)
plt.plot(A_GB - B_GB*GB_test, label='Green/Blue', color='limegreen')
plt.plot(SpO2_test, label='SpO2', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
ax = plt.subplot(6, 2, 10)
plt.plot(A_BR - B_BR*BR_test, label='Blue/Red', color='dodgerblue')
plt.plot(SpO2_test, label='SpO2', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
ax = plt.subplot(6, 2, 11)
plt.plot(A_BI - B_BI*BI_test, label='Blue/IR', color='dodgerblue')
plt.plot(SpO2_test, label='SpO2', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
ax = plt.subplot(6, 2, 12)
plt.plot(A_BG - B_BG*BG_test, label='Blue/Green', color='dodgerblue')
plt.plot(SpO2_test, label='SpO2', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
plt.show()

print('Red/IR to SpO2 - Error: ',np.mean(abs((A_R - B_R*R_test) - SpO2_test)),'%')
print('Red/Green to SpO2 - Error: ',np.mean(abs((A_RG - B_RG*RG_test) - SpO2_test)),'%')
print('Red/Blue to SpO2 - Error: ',np.mean(abs((A_RB - B_RB*RB_test) - SpO2_test)),'%')
print('IR/Red to SpO2 - Error: ',np.mean(abs((A_IR - B_IR*IR_test) - SpO2_test)),'%')
print('IR/Green to SpO2 - Error: ',np.mean(abs((A_IG - B_IG*IG_test) - SpO2_test)),'%')
print('IR/Blue to SpO2 - Error: ',np.mean(abs((A_IB - B_IB*IB_test) - SpO2_test)),'%')
print('Green/Red to SpO2 - Error: ',np.mean(abs((A_GR - B_GR*GR_test) - SpO2_test)),'%')
print('Green/IR to SpO2 - Error: ',np.mean(abs((A_GI - B_GI*GI_test) - SpO2_test)),'%')
print('Green/Blue to SpO2 - Error: ',np.mean(abs((A_GB - B_GB*GB_test) - SpO2_test)),'%')
print('Blue/Red to SpO2 - Error: ',np.mean(abs((A_BR - B_BR*BR_test) - SpO2_test)),'%')
print('Blue/IR to SpO2 - Error: ',np.mean(abs((A_BI - B_BI*BI_test) - SpO2_test)),'%')
print('Blue/Green to SpO2 - Error: ',np.mean(abs((A_BG - B_BG*BG_test) - SpO2_test)),'%')
'''

'''
# Calculate Coefficents
A_SVR,B_SVR = calc_coefficients_SVR(windowed_train_R, windowed_train_SpO2)
A_LR,B_LR = calc_coefficients_LR(windowed_train_R, windowed_train_SpO2)
A_HR,B_HR = calc_coefficients_HR(windowed_train_R, windowed_train_SpO2)
A_RANSAC,B_RANSAC = calc_coefficients_RANSAC(windowed_train_R, windowed_train_SpO2)
'''

## Plotting
'''
# Test results from different Regressors and SpO2
plt.plot(A_SVR - B_SVR*test_R, label='SVR')
plt.plot(A_LR - B_LR*test_R, label='LinearR')
plt.plot(A_HR - B_HR*test_R, label='HuberR')
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

# Red, IR, Blue, Green PPG signals 
plt.plot(blue, label='blue')
plt.plot(ir, label='ir')
plt.plot(green, label='green')
plt.plot(red, label='red')
plt.legend(loc='best')
plt.show()

# Squared Sum of acceleration 
plt.plot(new_acc, label='acc')
plt.legend(loc='best')
plt.show()
'''




