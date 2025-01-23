import numpy as np
import pickle
from init import *
from utils import *

if __name__ == '__main__':

    data= Data()  # create a parameter instance
    params = Params(data)
    lent = params.lent
    
    # #first data set: magnetic particle core size
    # dcore_list = np.array([20, 30, 40, 50, 60])*1e-9
    # dhyd_list = dcore_list + 10e-9
    # M = np.zeros( (len(dcore_list), lent) )
    # N = np.zeros( (len(dcore_list), lent) )
    # sigH = np.zeros( (len(dcore_list), lent) )
    # for i in range(len(dcore_list)):
    #     print(f"\ndcore={dcore_list[i]} ----- start", flush=True)
    #     data.dCore = dcore_list[i]
    #     data.dHyd = dhyd_list[i]
    #     M[i, :], N[i, :], sigH[i,:] = MPI_Langevin_std_core(data)
    #     print(f"\ndcore={dcore_list[i]} ----- completed.")
    # np.savetxt('MNP-properties/data/Core_M.csv', M, delimiter=',', comments='')
    # np.savetxt('MNP-properties/data/Core_N.csv', N, delimiter=',', comments='')
    # np.savetxt('MNP-properties/data/Core_sigH.csv', sigH, delimiter=',', comments='')

    # #second data set: magnetic particle hydrodynamic size
    # dco = 30e-9
    # data.dCore = dco
    # dhyd_list = np.array([30, 35, 40, 45, 55, 60])*1e-9
    # M = np.zeros( (len(dhyd_list), lent) )
    # N = np.zeros( (len(dhyd_list), lent) )
    # sigH = np.zeros( (len(dhyd_list), lent) )
    # for i in range(len(dhyd_list)):
    #     print(f"\ndhyd={dhyd_list[i]} ----- start", flush=True)
    #     data.dHyd = dhyd_list[i]
    #     M[i, :], N[i, :], sigH[i,:] = MPI_Langevin_std_core(data)
    #     print(f"\ndhyd={dhyd_list[i]} ----- completed.")
    # np.savetxt('MNP-properties/data/Hyd_M.csv', M, delimiter=',', comments='')
    # np.savetxt('MNP-properties/data/Hyd_N.csv', N, delimiter=',', comments='')
    # np.savetxt('MNP-properties/data/Hyd_sigH.csv', sigH, delimiter=',', comments='')

    # # third data set: magnetic particle anisotropy
    # dco = 30e-9
    # data.dCore = dco
    # dhyd = 40e-9
    # data.dHyd = dhyd
    # ka_list = np.array([3, 5, 8, 10, 15])*1e3
    # M = np.zeros( (len(ka_list), lent) )
    # N = np.zeros( (len(ka_list), lent) )
    # sigH = np.zeros( (len(ka_list), lent) )
    # for i in range(len(ka_list)):
    #     print(f"\nka={ka_list[i]} ----- start", flush=True)
    #     data.kAnis = ka_list[i]
    #     M[i, :], N[i, :], sigH[i,:] = MPI_Langevin_std_core(data)
    #     print(f"\nka={ka_list[i]} ----- completed.")
    # np.savetxt('MNP-properties/data/Anis_M.csv', M, delimiter=',', comments='')
    # np.savetxt('MNP-properties/data/Anis_N.csv', N, delimiter=',', comments='')
    # np.savetxt('MNP-properties/data/Anis_sigH.csv', sigH, delimiter=',', comments='')

    # forth data set: magnetic particle saturation magnetization
    dco = 30e-9
    data.dCore = dco
    dhyd = 40e-9
    data.dHyd = dhyd
    ka = 3e3
    data.kAnis = ka
    Ms_list = np.array([50, 100, 200, 300, 400, 500])*1e3
    M = np.zeros( (len(Ms_list), lent) )
    N = np.zeros( (len(Ms_list), lent) )
    sigH = np.zeros( (len(Ms_list), lent) )
    for i in range(len(Ms_list)):
        print(f"\nMs={Ms_list[i]} ----- start", flush=True)
        data.Ms = Ms_list[i]
        M[i, :], N[i, :], sigH[i,:] = MPI_Langevin_std_core(data)
        print(f"\nMs={Ms_list[i]} ----- completed.")
    np.savetxt('MNP-properties/data/Satu_M.csv', M, delimiter=',', comments='')
    np.savetxt('MNP-properties/data/Satu_N.csv', N, delimiter=',', comments='')
    np.savetxt('MNP-properties/data/Satu_sigH.csv', sigH, delimiter=',', comments='')


