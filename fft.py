import numpy as np
import pandas as pd
from Ollies_EMG_Data_Processing import import_data
from Ollies_EMG_Data_Processing.import_data import all_data, df_C, df_O, df_D, df_L,df_R,df_U
import matplotlib.pyplot as plt

np_all_data = import_data.all_data.to_numpy()

y= np.fft.fft(np_all_data[0, 3:])
#plt.plot(df_C.columns[3:], df_C.iloc[0, 3:])
#plt.plot(all_data.columns[3:], y)
#plt.show()


def get_features(s):

    np_s = s.to_numpy()

    features = np.empty(shape = (297,5) )

  
    for i in range(s.shape[0]):
        
        #mean of fft
        fft_arr = np.fft.fft(np_s[i, 3:]) 
        features[i,0] = np.mean(fft_arr)

        #variance
        features[i,1] = np.var(np_s[i, 3:])

        #standard deviation
        features[i,2] = np.std(np_s[i, 3:])

        #mean average value
        features[i,3] = np.mean(np_s[i, 3:])

        #energy
        


        
    print(features)


get_features(df_C)
    


