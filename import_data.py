import json
import pandas as pd
import numpy as np

# Read data
filepath = "/home/tnt/Cognimanus/legacy/EMG_hand_data.json"
with open(filepath) as json_file:
    data = json.load(json_file)

# Write quick function to read into 2D
# Will take a specific label return a 2D pandas dataframe
def load_2D(data, key):
    # Iterate through list elements to flatten
    l = []
    for s in data[key]['data']:
        l_ = []
        # Manually append first 3 elements
        l_.append(s[0])
        l_.append(s[1])
        l_.append(s[2])
        
        # Iterate through array element and append
        for d in s[3]:
            l_.append(d)
            
        # Finally append to L
        l.append(l_)
        
    # Also flatten header
    h = []
    head = data[key]['header']
    h.append(head[0])
    h.append(head[1])
    h.append(head[2])
    for t in head[3]:
        h.append(t)
        
    return pd.DataFrame(l, columns=h)



df_R = load_2D(data, 'R')
df_U = load_2D(data, 'U')
df_O = load_2D(data, 'O')
df_T = load_2D(data, 'T')
df_M = load_2D(data, 'M')
df_I = load_2D(data, 'I')
df_P = load_2D(data, 'P')

df_O = df_O.sample(frac = 1/6).reset_index(drop = True)

all_data = pd.concat([df_O, df_U, df_R, df_T, df_M, df_I, df_P], ignore_index=True)


    
