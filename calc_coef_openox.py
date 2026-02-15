import numpy as np
import sys
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from ppg_functions import get_R_values, get_SpO2_values, time_sync, data_shuffle


class coefficients():

    A : float
    B : float
    sample_len = 10
    window_len = 10
    ppg_fs = 86
    spo2_fs = 2
    sao2_fs = 2

    def __init__(self,files):
        R_values = np.array([0])
        SpO2_values = np.array([0])

        for filename in files:
            print(filename)
            current_R_values, ppg_base_time = get_R_values(filename+'_ppg', self.sample_len, self.ppg_fs)
            current_SpO2_values, SpO2_base_time = get_SpO2_values(filename+'_2hz.csv', self.sample_len, self.spo2_fs)
            synced_R_values, synced_SpO2_values = time_sync(current_R_values, ppg_base_time, current_SpO2_values, SpO2_base_time)
            
            
            R_values = np.vstack((R_values,synced_R_values))
            SpO2_values = np.hstack((SpO2_values,synced_SpO2_values))
        
        shuffled_R_values, shuffled_SpO2_values = data_shuffle(R_values[1:], SpO2_values[1:], self.window_len)
        #shuffled_R_values, shuffled_SpO2_values = R_values[1:], SpO2_values[1:]
        self.A, self.B = self.calc_coefficients(shuffled_R_values, shuffled_SpO2_values) 


    
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
    
