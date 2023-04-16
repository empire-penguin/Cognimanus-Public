import numpy as np
import pandas as pd
import import_data
from import_data import all_data, df_O, df_M, df_T,df_R,df_U, df_I, df_P
import matplotlib.pyplot as plt
import math
#the point of this file is to get the five features we want as inputs for our model:
# - fft decomposition(I did mean of fft though because that gives a single value and that's 
#   what the article used)
# - mean average value
# - variance
# - standard deviation
# - energy


def get_features(s):
    """
    gets the features we want from the data, as well as the labels of each sample

    for labels(Using Ollie's for now because we don't have our own data yet):
    - O = 0
    - C = 1
    - L = 2
    - R = 3
    - U = 4
    - D = 5
    """

    np_s = s.to_numpy()

    
    features = np.empty(shape = (np_s.shape[0],11+650+650), dtype = float)

  
    for i in range(np_s.shape[0]):

        #for labels
        if np_s[i,0] == 'O':
            features[i,0:7] = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0])
        if np_s[i,0] == 'U':
            features[i,0:7] = np.array([0.0,1.0,0.0,0.0,0.0,0.0,0.0])
        if np_s[i,0] == 'R':
            features[i,0:7] = np.array([0.0,0.0,1.0,0.0,0.0,0.0,0.0])
        if np_s[i,0] == 'T':
            features[i,0:7] = np.array([0.0,0.0,0.0,1.0,0.0,0.0,0.0])
        if np_s[i,0] == 'M':
            features[i,0:7] = np.array([0.0,0.0,0.0,0.0,1.0,0.0,0.0])
        if np_s[i,0] == 'I':
            features[i,0:7] = np.array([0.0,0.0,0.0,0.0,0.0,1.0,0.0])
        if np_s[i,0] == 'P':
            features[i,0:7] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0])
    
        np_s_i = np_s[i]
        #print(np_s_i.shape)
        #for j in range(np_s_i.shape[0]):
            #if j>0:
              #  if math.isnan(np_s_i[j]):
              #      np_s_i = np.delete(arr = np_s_i, obj = j)
               #     print(np_s_i.shape[0])

        j = 1
        while j <= np_s_i.shape[0]:
            if j >= np_s_i.shape[0]:
                break
            elif math.isnan(np_s_i[j]):
                np_s_i = np.delete(arr = np_s_i, obj = j)
                #print(np_s_i.shape[0])
            else:
                j+=1



        # fft( article uses mean of fft, so maybe consider that )
        fft_arr = np.fft.fft(np_s_i[3:]) 
        features[i,7:7+len(fft_arr)] = fft_arr
        #print(np.mean(fft_arr))

        #variance
        features[i,8+len(fft_arr)] = np.var(np_s_i[3:])
        #print(np.var(np_s_i[3:]))

        #standard deviation
        features[i,9+len(fft_arr)] = np.std(np_s_i[ 3:])
        #print(np.std(np_s_i[ 3:]))

        #mean average value
        features[i,10+len(fft_arr)] = np.mean(np_s_i[3:])
        #print(np.mean(np_s_i[3:]))

        #energy(still needs to be done)
        energy_k = 0
        energy = []
        for j in np_s_i[3:]:
            energy_k += j**2
            energy.append(energy_k)
        features[i,11+len(fft_arr):11+len(fft_arr)+len(energy)] = energy
        #print(len(energy))

    #features = np.hstack((np_s[:,0], features))
    print(features) 
    return features



get_features(all_data)
    


