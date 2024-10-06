import numpy as np
from init import *
from utlis import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    params = Params()  # create a parameter instance

    # core size 25
    core_size = 25e-9
    hyd_size = core_size + 10e-9
    params.set_params(dCore=core_size, dHyd=hyd_size)
    #minDistList = np.array([35, 60, 80, 100, 150, 200, 300, 500])
    minDistList = np.array([100])
    M = np.zeros( (len(minDistList), len(params.tu)) )
    Hdd = np.zeros(((len(minDistList), len(params.tu))))
    for i, d in enumerate(minDistList):
        params.set_params(minDist=d)
        init = params.get_params()
        M[i,:], Hdd[i] = CombinedNeelBrown(init_data = init)
        print('\r', 'coresize25_minDist: ' + "." * 10 + " ", end=str(i)+'/'+str(len(minDistList)-1))
    print()
    np.savetxt('CoreSize25.csv', M, delimiter=',', comments='')
    np.savetxt('ddField25.csv', Hdd, delimiter=',', comments='')

    # core size 30
    core_size = 30e-9
    hyd_size = core_size + 10e-9
    params.set_params(dCore=core_size, dHyd=hyd_size)
    minDistList = np.array([40, 60, 80, 100, 150, 200, 300, 500])
    M = np.zeros( (len(minDistList), len(params.tu)) )
    Hdd = np.zeros( (len(minDistList), len(params.tu)) )
    for i, d in enumerate(minDistList):
        params.set_params(minDist=d)
        init = params.get_params()
        M[i,:], Hdd[i] = CombinedNeelBrown(init_data = init)
        print('\r', 'coresize30_minDist: ' + "." * 10 + " ", end=str(i)+'/'+str(len(minDistList)-1))
    print()
    np.savetxt('CoreSize30.csv', M, delimiter=',', comments='')
    np.savetxt('ddField30.csv', Hdd, delimiter=',', comments='')


    # core size 35
    core_size = 35e-9
    hyd_size = core_size + 10e-9
    params.set_params(dCore=core_size, dHyd=hyd_size)
    minDistList = np.array([45, 60, 80, 100, 150, 200, 300, 500])
    M = np.zeros( (len(minDistList), len(params.tu)) )
    Hdd = np.zeros( (len(minDistList), len(params.tu)) )
    for i, d in enumerate(minDistList):
        params.set_params(minDist=d)
        init = params.get_params()
        M[i,:], Hdd[i] = CombinedNeelBrown(init_data = init)
        print('\r', 'coresize35_minDist: ' + "." * 10 + " ", end=str(i)+'/'+str(len(minDistList)-1))
    print("\n")
    np.savetxt('CoreSize35.csv', M, delimiter=',', comments='')
    np.savetxt('ddField35.csv', Hdd, delimiter=',', comments='')