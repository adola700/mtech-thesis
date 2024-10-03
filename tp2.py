############# basic imports, experiment and device init #################
import matplotlib.pyplot as plt
import numpy as np
import os
# import cv2
import math
import h5py
import time
# import torch
# import torch.nn as nn
from PhysNetModel import PhysNet
from loss import ContrastLoss,CalculateNormPSD,SupervisedLoss
from IrrelevantPowerRatio import IrrelevantPowerRatio
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from utils_data import H5Dataset, UBFC_LU_split
from utils_sig import hr_fft, butter_bandpass, hr_from_PSD
# from torch import optim
# from torch.utils.data import DataLoader
# from sacred import Experiment
# from sacred.observers import FileStorageObserver
from helper_fns import EarlyStopper, standardize_and_filter, load_dict, Plotter,standardize
from imports_and_utils import clean_bvp, only_filter
# ex = Experiment('model_train', save_git_info=False)
import h5py
from PhysNetModel import PhysNet
# from loss import ContrastLoss
# from IrrelevantPowerRatio import IrrelevantPowerRatio
# import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from torch import optim
# from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import KFold, train_test_split
from utils_data import H5Dataset, UBFC_LU_split
from utils_sig import get_best_hr_2000, get_best_hr
from helper_fns import EarlyStopper, standardize_and_filter, load_dict
from finetune_ubfc import finetune
from scipy import signal
import scipy
# import warnings
# warnings.filterwarnings("error")

import sys
# # ## CONFIGS
# if torch.cuda.is_available():
#     device = torch.device("cuda:2")
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#     print("running using gpu !")
# else:
#     device = torch.device('cpu')
import warnings
# with warnings.catch_warnings(record=True) as caught_warnings:
#     warnings.filterwarnings("error", category=UserWarning)
# warnings.filterwarnings("error", category=UserWarning)


# ##dl model for inferencing
# def dl_model(model, imgs_clip):
#         model.eval()
#         # model inference
#         img_batch = imgs_clip
#         img_batch = img_batch.transpose((3,0,1,2))
#         img_batch = img_batch[np.newaxis].astype('float32')
#         img_batch = torch.tensor(img_batch).to(device)
#         with torch.no_grad():
#             rppg = model(img_batch)[:,-1, :]
#             rppg = rppg[0].detach().cpu().numpy()
#         return rppg


# Load Model

# model = PhysNet(S=2, in_ch=3).to(device)
# if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model, device_ids = [2,3])
#         model.to(device)
# else:
#         model.to(device)

# weights_path = "./cv_results/98/epoch2__rep_0_fold_2.pt"
# model.eval()
# model.load_state_dict(torch.load(weights_path))
h5_dir = '../h5_files'
all_paths = [h5_dir + '/' + f"{i}.h5" for i in range(1,50) if  f"{i}.h5" in os.listdir(h5_dir)]
path = f"../h5_files/22.h5"

hrs = []
hrs_2000 = []
# testing


for path in [f"../h5_files/20.h5","../h5_files/23.h5"]:
    with h5py.File(path, 'r') as f:
        imgs = f['imgs']
        bvp = f['bvp']
        duration = bvp.shape[0] / 30
        num_blocks = int(duration // 30)
        for b in range(num_blocks):
            # b = 0
            fs = 30
            time_interval = 30
            bvp_clip = bvp[b*time_interval*fs:(b+1)*time_interval*fs]
            # try:
            # warnings.warn("This is a warning", DeprecationWarning)
            # with warnings.catch_warnings(record=True) as caught_warnings:
            #     # warnings.filterwarnings("error", category=UserWarning)
            hr = get_best_hr(bvp_clip)
            print(f"path:{path}, hr:{hr}")        

                # if caught_warnings:
                    # Raise an exception to handle the captured warnings
                    # print("ffffff")
            # except UserWarning as warning:
            #     print("Caught UserWarning:", str(warning))

            
            # except Warning:
                # print("A warning occurred. Exiting the program.")
                # sys.exit()
            # print(f"path:{path}, hr_2000:{get_best_hr_900(only_filter(bvp_clip))}")
            # hrs.append(get_best_hr_900(bvp_clip))
            # hrs_2000.append(get_best_hr_900(only_filter(bvp_clip)))
            # if(abs(hrs[-1] - hrs_2000[-1]) >0.6):
                #  print("\tOUTLIER")
                 
            # rppg_clip = dl_model(model,imgs[b*time_interval*fs:(b+1)*time_interval*fs])
            # print(mean_absolute_error(bvp_clip, rppg_clip)**0.5)
            # bvp_psd = get_psd(bvp_clip)
            # rppg_psd = get_psd(rppg_clip)

            # fig, axs = plt.subplots(3)
            # # freqs = np.linspace(36.0,240.0,len(bvp_psd))

            # # print("max for 500fps",freqs[np.argmax(sig_psd)])   
            # # print("max for 500fps",freqs[np.argmax(bvp_psd)])   

            # axs[0].plot(np.linspace(0,30,len(bvp_psd)), bvp_psd, label = "bvp")
            # axs[0].plot(np.linspace(0,30,len(bvp_clip)), bvp_clip)
            # axs[0].set_title("bvp_before cleaning")
            # axs[0].plot(np.linspace(0,30,len(bvp_clip)), bvp_clip,  label = "bvp")
            # axs[0].plot(np.linspace(0,30,len(standardize_and_filter(bvp_clip))), standardize(rppg_clip),  label = "clean")
            # axs[1].plot(np.linspace(0,30,len(standardize_and_filter(bvp_clip))), only_filter(rppg_clip),  label = "only filter")
            # axs[2].plot(np.linspace(0,30,len(standardize_and_filter(bvp_clip))), standardize(bvp_clip),  label = "raw bvp")
            # plt.legend()
            # plt.savefig("./bvp.png")
            # plt.close()
            # # print(get_best_hr(standardize(clean_bvp(bvp_clip))))
            # # print(get_best_hr(standardize_and_filter(bvp_clip)))
            # print("bvp",get_best_hr(bvp_clip))
            # print("rppg",get_best_hr(rppg_clip))
            # print("rppg",get_best_hr(only_filter(rppg_clip)))
            # # axs[1].plot(np.linspace(0,30,len(bvp_clip)), clean_bvp(bvp_clip))
            # # axs[1].set_title("rppg")
            # # axs[1].set_title("bvp after cleaning")

            
            # # # print("PADDED RES",(240-36)/len(bvp_psd))
            # # print("hr_fft bvp_clip",hr_fft(bvp_clip,30,zero_pad=100))
            # # print("hr_from_PSD bvp_clip",hr_from_PSD(bvp_clip,30,zero_pad=100))

            # # print("hr_fft clean_bvp",hr_fft(clean_bvp(bvp_clip),30,zero_pad=100))
            # # print("hr_from_PSD clean_bvp",hr_from_PSD(clean_bvp(bvp_clip),30,zero_pad=100))
            
            # # # fig, axs = plt.subplots(2)

            # wd, m = hp.process((rppg_clip), sample_rate = 30.0)
            # print("rppg pred", get_best_hr(rppg_clip,30.0))

            # hp.plotter(wd, m)
            # plt.savefig("./hpy1.png")
            # plt.close()

            # wd, m1 = hp.process(clean_bvp(rppg_clip), sample_rate = 30.0)
            # print("bvp pred",get_best_hr((bvp_clip),30))
            # hp.plotter(wd, m1)
            # plt.savefig("./hpy2.png")

# print("mae: ",mean_absolute_error(hrs, hrs_2000))
# import pandas as pd
# print(pd.Series([x - y for x, y in zip(hrs, hrs_2000)]).describe())