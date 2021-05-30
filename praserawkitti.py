# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:22:33 2020

@author: Aamanku
"""

import numpy as np
import os
dir_data='dataset/data/' #location of data txt files for sensors
timestamps_file='dataset/timestamps.txt' #and their timestamps

v_forward=list()
v_leftward=list()
for filename in os.listdir(dir_data):
    # print(filename)
    with open(dir_data+filename,'r') as f:
        read_data=f.readlines()
        data=str(read_data).split(' ')
        v_forward.append(data[8])
        v_leftward.append(data[9])
v_forward=np.array(v_forward,dtype=np.float32)
v_leftward=np.array(v_leftward,dtype=np.float32)

timestamps=list()
with open(timestamps_file) as ts:
    for stamp in ts:
        data=np.float32(str(stamp).split(':')[-1])+np.float32(str(stamp).split(':')[-2])*60
        timestamps.append(data)

timestamps=np.float32(timestamps)
data_final=np.stack((timestamps,v_forward,v_leftward),1)
np.savetxt('data_kitti.txt',data_final)






