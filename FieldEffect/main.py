import numpy as np
from init import *
from utils import *

if __name__ == '__main__':

    data = Data() 

    f = data.fieldFreq
    Ms = data.Ms
    cycs = data.nPeriod
    num = data.nParticle
    fieldAml_list = 1e-3*np.array([20, 15, 10, 5]) 

        # data set: magnetic particle core size = 25 nm
    dco = 20e-9
    dhyd = dco+10e-9
    rsol = 15000

    data.dCore = dco
    data.dHyd = dhyd
    data.rsol = rsol

    params = MPI_Langevin_std_init(data)

    dt = 1/params.fs
    if dt >= params.t0:
      print(f"Warning: time step dt={dt} is longer than Neel attemping time tau0={params.t0} that can be the source of inaccuray")

    tf = params.tf
    lent = params.lent
    t = np.array([i*dt for i in range(lent)])
    M = np.zeros( (len(fieldAml_list), lent) )
    N = np.zeros( (len(fieldAml_list), lent) )
    sigH = np.zeros( (len(fieldAml_list), lent) )
    print(f"simulation for dco = {dco} is started", flush=True)
    for i, a in enumerate(fieldAml_list):
        data.fieldAmpl = a
        M[i, :], N[i, :], sigH[i,:] = MPI_Langevin_std_core(data)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(fieldAml_list)-1))
    print(f"\nsimulation for dco = {dco} is done", flush=True)
    np.savetxt('FieldEffect/data/Core20_M.csv', M, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core20_N.csv', N, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core20_sigH.csv', sigH, delimiter=',', comments='')

    # data set: magnetic particle core size = 25 nm
    dco = 25e-9
    dhyd = dco+10e-9

    data.dCore = dco
    data.dHyd = dhyd
    data.rsol = rsol

    params = MPI_Langevin_std_init(data)

    dt = 1/params.fs
    if dt >= params.t0:
      print(f"Warning: time step dt={dt} is longer than Neel attemping time tau0={params.t0} that can be the source of inaccuray")

    tf = params.tf
    lent = params.lent
    t = np.array([i*dt for i in range(lent)])
    M = np.zeros( (len(fieldAml_list), lent) )
    N = np.zeros( (len(fieldAml_list), lent) )
    sigH = np.zeros( (len(fieldAml_list), lent) )
    print(f"simulation for dco = {dco} is started", flush=True)
    for i, a in enumerate(fieldAml_list):
        data.fieldAmpl = a
        M[i, :], N[i, :], sigH[i,:] = MPI_Langevin_std_core(data)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(fieldAml_list)-1))
    print(f"\nsimulation for dco = {dco} is done", flush=True)
    np.savetxt('FieldEffect/data/Core25_M.csv', M, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core25_N.csv', N, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core25_sigH.csv', sigH, delimiter=',', comments='')

    # data set: magnetic particle core size = 30 nm
    dco = 30e-9
    dhyd = dco+10e-9

    data.dCore = dco
    data.dHyd = dhyd

    params = MPI_Langevin_std_init(data)

    dt = 1/params.fs
    if dt >= params.t0:
        print(f"Warning: time step dt={dt} is longer than Neel attemping time tau0={params.t0} that can be the source of inaccuray")
    tf = params.tf
    lent = params.lent
    t = np.array([i*dt for i in range(lent)])
    M = np.zeros( (len(fieldAml_list), lent) )
    N = np.zeros( (len(fieldAml_list), lent) )
    sigH = np.zeros( (len(fieldAml_list), lent) )
    print(f"simulation for dco = {dco} is started", flush=True)
    for i, a in enumerate(fieldAml_list):
        data.fieldAmpl = a
        M[i, :], N[i, :], sigH[i,:] = MPI_Langevin_std_core(data)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(fieldAml_list)-1))
    print(f"\nsimulation for dco = {dco} is done", flush=True)
    np.savetxt('FieldEffect/data/Core30_M.csv', M, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core30_N.csv', N, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core30_sigH.csv', sigH, delimiter=',', comments='')

    # data set: magnetic particle core size = 35 nm
    dco = 35e-9
    dhyd = dco+10e-9

    data.dCore = dco
    data.dHyd = dhyd

    params = MPI_Langevin_std_init(data)

    dt = 1/params.fs
    if dt >= params.t0:
        print(f"Warning: time step dt={dt} is longer than Neel attemping time tau0={params.t0} that can be the source of inaccuray")
    tf = params.tf
    lent = params.lent
    M = np.zeros( (len(fieldAml_list), lent) )
    N = np.zeros( (len(fieldAml_list), lent) )
    sigH = np.zeros( (len(fieldAml_list), lent) )
    print(f"simulation for dco = {dco} is started", flush=True)
    for i, a in enumerate(fieldAml_list):
        data.fieldAmpl = a
        M[i,:], N[i, :], sigH[i,:] = MPI_Langevin_std_core(data)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(fieldAml_list)-1))
    print(f"\nsimulation for dco = {dco} is done", flush=True)
    np.savetxt('FieldEffect/data/Core35_M.csv', M, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core35_N.csv', N, delimiter=',', comments='')
    np.savetxt('FieldEffect/data/Core35_sigH.csv', sigH, delimiter=',', comments='')
