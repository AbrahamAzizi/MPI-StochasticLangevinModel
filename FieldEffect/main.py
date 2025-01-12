import numpy as np
from init import *
from utlis import *

if __name__ == '__main__':

    data = Data() 
    f = data.fieldFreq
    Ms = data.Ms
    cycs = data.nPeriod
    data.rsol = 100
    rsol = data.rsol
    num = data.nParticle
    fs = rsol*2*f
    dt = 1/fs
    tf = cycs*(1/f)
    lent = int(np.ceil(tf/dt))
    t = np.array([i*dt for i in range(lent)])

    # data set: magnetic particle core size = 25 nm
    data.dCore = 25e-9
    data.dHyd = data.dCore + 10e-9
    fieldAml_list = 1e-3*np.array([20, 15, 10, 5])
    M = np.zeros( (len(fieldAml_list), lent) )
    N = np.zeros( (len(fieldAml_list), lent) )
    sigH = np.zeros( (len(fieldAml_list), lent) )
    for i, a in enumerate(fieldAml_list):
        data.fieldAmpl = a
        M[i, :], N[i, :], sigH[i,:] = CombinedNeelBrown(data)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(fieldAml_list)-1))
    print()
    np.savetxt('FieldEffect/data/Core25_M.csv', M, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core25_N.csv', N, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core25_sigH.csv', sigH, delimiter=',', comments='')

    # data set: magnetic particle core size = 30 nm
    data.dCore = 30e-9
    data.dHyd = data.dCore + 10e-9
    fieldAml_list = 1e-3*np.array([20, 15, 10, 5])
    M = np.zeros( (len(fieldAml_list), lent) )
    N = np.zeros( (len(fieldAml_list), lent) )
    sigH = np.zeros( (len(fieldAml_list), lent) )
    for i, a in enumerate(fieldAml_list):
        data.fieldAmpl = a
        M[i, :], N[i, :], sigH[i,:] = CombinedNeelBrown(data)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(fieldAml_list)-1))
    print()
    np.savetxt('FieldEffect/data/Core30_M.csv', M, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core30_N.csv', N, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core30_sigH.csv', sigH, delimiter=',', comments='')

    # data set: magnetic particle core size = 35 nm
    data.dCore = 35e-9
    data.dHyd = data.dCore + 10e-9
    fieldAml_list = 1e-3*np.array([20, 15, 10, 5])
    M = np.zeros( (len(fieldAml_list), lent) )
    N = np.zeros( (len(fieldAml_list), lent) )
    sigH = np.zeros( (len(fieldAml_list), lent) )
    for i, a in enumerate(fieldAml_list):
        data.fieldAmpl = a
        M[i,:], N[i, :], sigH[i,:] = CombinedNeelBrown(data)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(fieldAml_list)-1))
    print()
    np.savetxt('FieldEffect/data/Core35_M.csv', M, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core35_N.csv', N, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core35_sigH.csv', sigH, delimiter=',', comments='')
