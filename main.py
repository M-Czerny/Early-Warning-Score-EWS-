import numpy as np
import pandas as pd
from calc_coef import coefficients
from validation import validate
import os
from pathlib import Path
import wfdb

path_files = []
ppg_files = []
pathstring = 'data/waveforms/'
for x in range(0,1):
    pathlist = Path(pathstring + '{folder}/'.format(folder = x)).glob('**/*_ppg.dat')
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)
        if Path.is_file(path_in_str[0:-8]+'_2hz.csv'):
            ppg_files.append(os.path.basename(path_in_str)[0:-8])   
            path_files.append(path_in_str[0:-8])   
    
participants = []
data = []
for file in ppg_files:
    df = pd.read_csv('data/encounter.csv')
    devices = df.loc[df['encounter_id'] == file, ['right_ear_device','left_ear_device','forehead_device','finger_l1_device','finger_l2_device','finger_l3_device','finger_l4_device','finger_l5_device','finger_r1_device','finger_r2_device','finger_r3_device','finger_r4_device','finger_r5_device']].values[0]
    participant = df.loc[df['encounter_id'] == file, 'patient_id'].values[0]
    if np.isnan(devices).all():
        continue
    elif np.isnan(devices[0:2]).all():
        data.append([file, participant, 'fingers'])
        participants.append(participant)
    elif ~np.isnan(devices[0:1]).all():
        data.append([file, participant, 'ear'])
        participants.append(participant)
    elif ~np.isnan(devices[2]).all():
        data.append([file, participant, 'chest'])
        participants.append(participant)

#print(data)
    #print(devices,participant)

coef = coefficients(path_files[0:2])


test_file = 'data/waveforms/1/1ab221ec6347760891fcad58715195d06379a8b95db4a143eb61de8184af4cc9'

wfdb.plot_wfdb(wfdb.rdrecord(test_file+'_ppg', channels=[0,4]))

if test_file in participants:
    print('participant included in training')
else:
    validate(coef.A,coef.B,test_file)