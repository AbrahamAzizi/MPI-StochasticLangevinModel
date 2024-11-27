import numpy as np
from init import *
from utlis import *

if __name__ == '__main__':
    
    data = Data() 
    f = data.fieldFreq
    cycs = data.nPeriod
    data.rsol = 100
    rsol = data.rsol
    fs = rsol*2*f
    dt = 1/fs
    tf = cycs*(1/f)
    lent = int(np.ceil(tf/dt))

    # IGP30
    meanCoreIGP30 = 29.53*1e-9
    data.meanCore = meanCoreIGP30
    coatingSize = 16*1e-9
    data.coatingSize = coatingSize    
    stdCoreIGP30 = .09
    stdCore_list = np.round(stdCoreIGP30*np.array([.5, 1, 3, 5, 10]),3)
    M = np.zeros( (len(stdCore_list), lent) )
    for i, d in enumerate(stdCore_list):
        data.stdCore = d
        M[i, :], _, svIPG30= CombinedNeelBrown(data)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(stdCore_list)-1))
    print()
    np.savetxt('MNP-sizeDistribution/data/IPG30.csv', M, delimiter=',', comments='')

    # SHS30
    meanCoreSHS30 = 29.52*1e-9
    data.meanCore = meanCoreSHS30
    coatingSize = 28*1e-9
    data.coatingSize = coatingSize 
    stdCoreSHS30 = .08
    stdCore_list = np.round(stdCoreSHS30*np.array([.5, 1, 3, 5, 10]),3)
    M = np.zeros( (len(stdCore_list), lent) )
    for i, d in enumerate(stdCore_list):
        data.stdCore = d
        M[i, :], _, svSHS30 = CombinedNeelBrown(data)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(stdCore_list)-1))
    print()
    np.savetxt('MNP-sizeDistribution/data/SHS30.csv', M, delimiter=',', comments='')

    # SHP25
    meanCoreSHP25 = 25.65*1e-9
    data.meanCore = meanCoreSHP25
    coatingSize = 16*1e-9
    data.coatingSize = coatingSize 
    stdCoreSHP25 = .05
    stdCore_list = np.round(stdCoreSHP25*np.array([.5, 1, 3, 5, 10]),3)
    M = np.zeros( (len(stdCore_list), lent) )
    for i, d in enumerate(stdCore_list):
        data.stdCore = d
        M[i, :], _, svSHP25 = CombinedNeelBrown(data)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(stdCore_list)-1))
    print()
    np.savetxt('MNP-sizeDistribution/data/SHP25.csv', M, delimiter=',', comments='')

    # SHP15
    meanCoreSHP15 = 12.42*1e-9
    data.meanCore = meanCoreSHP15
    coatingSize = 16*1e-9
    data.coatingSize = coatingSize 
    stdCoreSHP15 = .11
    stdCore_list = np.round(stdCoreSHP15*np.array([.5, 1, 3, 5, 10]),3)
    M = np.zeros( (len(stdCore_list), lent) )
    for i, d in enumerate(stdCore_list):
        data.stdCore = d
        M[i, :], _, svSHP15 = CombinedNeelBrown(data)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(stdCore_list)-1))
    print()
    np.savetxt('MNP-sizeDistribution/data/SHP15.csv', M, delimiter=',', comments='')

    np.savetxt('MNP-sizeDistribution/data/sumVc.csv', np.array([svIPG30, svSHS30, svSHP25, svSHP15]), delimiter=',', comments='')

