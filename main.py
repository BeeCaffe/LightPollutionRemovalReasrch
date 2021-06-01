from src.tools.Utils import *
import numpy as np
import time
weight_save_dir = r'F:\tps_weight/'
row = 2048
col = 2048

mat = np.loadtxt(r'F:\tps_weight/w_0x31.txt')
st = time.time()
for i in range(row):
    for j in range(col):
        w_file_name = weight_save_dir + 'w_' + str(i) + 'x' + str(j) + '.txt'
        np.savetxt(w_file_name, mat)
        process('compute: ', i*col+j, row*col, st, time.time())
print('Done!')
