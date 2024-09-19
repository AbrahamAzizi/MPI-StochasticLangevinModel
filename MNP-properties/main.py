import numpy as np
from init import *
from utlis import *
import csv

if __name__ == '__main__':

    params = Params()  # create a parameter instance

    #first data set: magnetic particle core size
    d_core_list = np.array([20, 30, 40, 50, 60])
    d_hyd_list = d_core_list + 10
    Mz = []
    for i, d in enumerate(zip(d_core_list, d_hyd_list)):
        params.set_params(dCore = d[0]*1e-9, dHyd = d[1]*1e-9)
        init = params.get_params()
        M, _ = CombinedNeelBrown(init_data = init)
        Mz.append(M)
        print('\r', 'dcore_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(d_core_list)-1))
    print()
    with open('effectOfDcore.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(Mz)

    #second data set: magnetic particle hydrodynamic size
    d_core = 30e-9
    params.set_params(dCore = d_core)
    d_hyd_list = np.array([30, 35, 40, 45, 55, 60])
    Mz = []
    for i, d_hyd in enumerate(d_hyd_list):
        params.set_params(dHyd = d_hyd*1e-9)
        init = params.get_params()
        M, _ = CombinedNeelBrown(init_data = init)
        Mz.append(M)
        print('\r', 'dhyd_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(d_hyd_list)-1))
    print()
    with open('effectOfDhydrodynamic.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(Mz)

    # third data set: magnetic particle anisotropy
    d_core = 30e-9
    d_hyd = 40e-9
    params.set_params(dCore = d_core, dHyd = d_hyd)
    ka_list = np.array([3, 5, 8, 10, 15])
    Mz=[]
    for i, ka in enumerate(ka_list):
        params.set_params(kAnis = ka*1e3)
        init = params.get_params()
        M, _ = CombinedNeelBrown(init_data = init)
        Mz.append(M)
        print('\r', 'ka_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(ka_list)-1))
    print()
    with open('effectOfAnisotropy.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(Mz)

    # forth data set: magnetic particle saturation magnetization
    d_core = 30e-9
    d_hyd = 40e-9
    ka = 3e3
    params.set_params(dCore = d_core, dHyd = d_hyd, kAnis = ka)
    Ms_list = np.array([50, 100, 200, 300, 400, 500])
    Mz=[]
    for i, ms in enumerate(Ms_list):
        params.set_params(Ms = ms*1e3)
        init = params.get_params()
        M, _ = CombinedNeelBrown(init_data= init)
        Mz.append(M)
        print('\r', 'Ms_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(Ms_list)-1))
    print()
    with open('effectOfMagnetization.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(Mz)


