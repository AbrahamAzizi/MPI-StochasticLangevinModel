import numpy as np
from init import *
from utlis import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    
    data = Data()
    data.rsol = 100
    rsol = data.rsol # 20 for 1 MHz
    print("number of samples per period: ", rsol)
    B = data.fieldAmpl
    print("applied field amplitudes: ", B)
    f = data.fieldFreq
    print("applied field frequencies: ", f)
    cycs = data.nPeriod
    rsol = data.rsol

    fs = rsol*2*f
    print("sampling frequency: fs= ", fs)
    dt = 1/fs
    print("time step: dt= ", dt)
    tf = cycs*(1/f)
    print("final simulation time: tf= ", tf)
    lent = int(np.ceil(tf/dt))

    # core size 25
    core_size = 25e-9
    hyd_size = core_size + 10e-9
    data.dCore = core_size
    data.dHyd = hyd_size
    minDistList = np.array([50, 65, 100, 125, 150, 175, 200, 250])
    M = np.zeros( (len(minDistList), lent) )
    Hdd = np.zeros( (len(minDistList), lent) )
    for i, d in enumerate(minDistList):
        data.minDist = d
        M[i,:], Hdd[i, :] = CombinedNeelBrown(data)
        print('\r', 'coresize25_minDist: ' + "." * 10 + " ", end=str(i)+'/'+str(len(minDistList)-1))
    print()
    np.savetxt('dipoleIntractionEffect/data/M_size25.csv', M, delimiter=',', comments='')
    np.savetxt('dipoleIntractionEffect/data/Hdd_size25.csv', Hdd, delimiter=',', comments='')

    # core size 30
    core_size = 30e-9
    hyd_size = core_size + 10e-9
    data.dCore = core_size
    data.dHyd = hyd_size
    minDistList = np.array([55, 70, 100, 125, 150, 175, 200, 250])
    M = np.zeros( (len(minDistList), lent) )
    Hdd = np.zeros( (len(minDistList), lent) )
    for i, d in enumerate(minDistList):
        data.minDist = d
        M[i,:], Hdd[i, :] = CombinedNeelBrown(data)
        print('\r', 'coresize30_minDist: ' + "." * 10 + " ", end=str(i)+'/'+str(len(minDistList)-1))
    print()
    np.savetxt('dipoleIntractionEffect/data/M_size30.csv', M, delimiter=',', comments='')
    np.savetxt('dipoleIntractionEffect/data/Hdd_size30.csv', Hdd, delimiter=',', comments='')


    # core size 35
    core_size = 35e-9
    hyd_size = core_size + 10e-9
    data.dCore = core_size
    data.dHyd = hyd_size
    minDistList = np.array([60, 75, 100, 125, 150, 175, 200, 250])
    M = np.zeros( (len(minDistList), lent) )
    Hdd = np.zeros( (len(minDistList), lent) )
    for i, d in enumerate(minDistList):
        data.minDist = d
        M[i,:], Hdd[i, :] = CombinedNeelBrown(data)
        print('\r', 'coresize35_minDist: ' + "." * 10 + " ", end=str(i)+'/'+str(len(minDistList)-1))
    print("\n")
    np.savetxt('dipoleIntractionEffect/data/M_size35.csv', M, delimiter=',', comments='')
    np.savetxt('dipoleIntractionEffect/data/Hdd_size35.csv', Hdd, delimiter=',', comments='')
