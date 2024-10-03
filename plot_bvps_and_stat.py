import h5py
import numpy as np
import os
import heartpy as hp
import matplotlib.pyplot as plt
from utils_sig import butter_bandpass, hr_fft, hr_from_PSD
from sklearn.metrics import mean_absolute_error, mean_squared_error
from imports_and_utils import clean_bvp, only_filter
import glob,sys

h5_dir = '../akhila_data_new_h5_files'
all_paths = glob.glob(h5_dir + '/*')

# print(all_paths)

# sys.exit()
# time_interval = 30
# fs = 30

# actual_hrs = []
# # approx_hrs = []
for path in ["../akhila_data_h5_files/16_7.h5"]:
    with h5py.File(path, 'r') as f:
        imgs = f['imgs']
        bvp = f['bvp']
        bvp_clip = bvp[:]
        # duration = np.min([imgs.shape[0], bvp.shape[0]]) / fs
        # plt.plot(list(range(len(bvp))), bvp_clip)
        wd, m = hp.process(only_filter(bvp_clip,500), sample_rate = 500)
        hp.plotter(wd,m)

        plt.savefig("./bad_bvp.png")
#         num_blocks = int(duration // time_interval)

#         for b in range(num_blocks):
#             # rppg_clip = dl_model(imgs[b*time_interval*fs:(b+1)*time_interval*fs])
#             bvp_clip = bvp[b*time_interval*fs:(b+1)*time_interval*fs]
#             _, m = hp.process(only_filter(bvp_clip), sample_rate = 30)
#             #display measures computed
#             # for measure in m.keys():
#             actual_hrs.append(m["bpm"])
#             # print('%s: %f' %(measure, m[m.keys()[0]]))
#             _, m = hp.process(clean_bvp(bvp_clip), sample_rate = 30)
#             approx_hrs.append(m["bpm"])
#             print("path",path,"b",b,"actual:", actual_hrs[-1], "approx:",approx_hrs[-1])

# print(f"\nmae:{mean_absolute_error(actual_hrs, approx_hrs)}, rmse: {mean_squared_error(actual_hrs,approx_hrs)**0.5}")