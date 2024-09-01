import numpy as np
from init import *
from utlis import *

if __name__ == '__main__':
    params = Params()  # create a parameter instance

    # IGP30
    meanCoreIGP30 = 29.53*1e-9
    stdCoreIGP30 = .09
    coatingSize = 16*1e-9
    stdCore_list = np.round(stdCoreIGP30*np.array([.5, 1, 3, 5, 10]),3)
    ka = 3e3
    Ms = 3e5
    params.set_params(Ms = Ms, kAnis = ka, dCoreMean=meanCoreIGP30, coatingSize=coatingSize)
    M = np.zeros( (len(stdCore_list), len(params.tu)) )
    for i, d in enumerate(stdCore_list):
        params.set_params(dCoreStd = d)
        init = params.get_params()
        M[i,:], _ = CombinedNeelBrown(init_data = init)
        print('\r', 'stdCore_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(stdCore_list)-1))
    print()
    np.savetxt('IPG30.csv', M, delimiter=',', comments='')

    # SHS30
    meanCoreSHS30 = 29.52*1e-9
    stdCoreSHS30 = .08
    coatingSize = 28*1e-9
    stdCore_list = np.round(stdCoreSHS30*np.array([.5, 1, 3, 5, 10]),3)
    params.set_params(dCoreMean=meanCoreSHS30, coatingSize=coatingSize)
    M = np.zeros( (len(stdCore_list), len(params.tu)) )
    for i, d in enumerate(stdCore_list):
        params.set_params(dCoreStd = d)
        init = params.get_params()
        M[i,:], _ = CombinedNeelBrown(init_data = init)
        print('\r', 'stdCore_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(stdCore_list)-1))
    print()
    np.savetxt('SHS30.csv', M, delimiter=',', comments='')

    # SHP25
    meanCoreSHP25 = 25.65*1e-9
    stdCoreSHP25 = .05
    coatingSize = 16*1e-9
    stdCore_list = np.round(stdCoreSHP25*np.array([.5, 1, 3, 5, 10]),3)
    params.set_params(dCoreMean=meanCoreSHP25, coatingSize=coatingSize)
    M = np.zeros( (len(stdCore_list), len(params.tu)) )
    for i, d in enumerate(stdCore_list):
        params.set_params(dCoreStd = d)
        init = params.get_params()
        M[i,:], _ = CombinedNeelBrown(init_data = init)
        print('\r', 'stdCore_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(stdCore_list)-1))
    print()
    np.savetxt('SHP25.csv', M, delimiter=',', comments='')

    # SHP25
    meanCoreSHP15 = 12.42*1e-9
    stdCoreSHP15 = .11
    coatingSize = 16*1e-9
    stdCore_list = np.round(stdCoreSHP15*np.array([.5, 1, 3, 5, 10]),3)
    params.set_params(dCoreMean=meanCoreSHP15, coatingSize=coatingSize)
    M = np.zeros( (len(stdCore_list), len(params.tu)) )
    for i, d in enumerate(stdCore_list):
        params.set_params(dCoreStd = d)
        init = params.get_params()
        M[i,:], _ = CombinedNeelBrown(init_data = init)
        print('\r', 'stdCore_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(stdCore_list)-1))
    print()
    np.savetxt('SHP15.csv', M, delimiter=',', comments='')
