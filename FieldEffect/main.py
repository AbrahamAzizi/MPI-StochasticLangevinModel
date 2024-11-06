import numpy as np
from init import *
from utlis import *

if __name__ == '__main__':

    params = Params()  # create a parameter instance

    # data set: magnetic particle core size = 25 nm
    dcore = 25e-9
    dhyd = dcore + 10e-9
    fieldAml_list = 1e-3*np.array([20, 15, 10, 5])
    params.set_params(dCore=dcore, dHyd=dhyd)
    print(params.get_params(), end='\n')
    M = np.zeros( (len(fieldAml_list), len(params.tu)) )
    N = np.zeros( ( len(fieldAml_list), len(params.tu)))
    for i, d in enumerate(fieldAml_list):
        params.set_params(fieldB=d)
        init = params.get_params()
        M[i,:], N[i, :] = CombinedNeelBrown(init_data = init)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(fieldAml_list)-1))
    print()
    np.savetxt('FieldEffect/data/size25.csv', M, delimiter=',', comments='')

    # data set: magnetic particle core size = 30 nm
    dcore = 30e-9
    dhyd = dcore + 10e-9
    fieldAml_list = 1e-3*np.array([20, 15, 10, 5])
    params.set_params(dCore=dcore, dHyd=dhyd)
    print(params.get_params(), end='\n')
    M = np.zeros( (len(fieldAml_list), len(params.tu)) )
    N = np.zeros( ( len(fieldAml_list), len(params.tu)))
    for i, d in enumerate(fieldAml_list):
        params.set_params(fieldB=d)
        init = params.get_params()
        M[i,:], N[i, :] = CombinedNeelBrown(init_data = init)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(fieldAml_list)-1))
    print()
    np.savetxt('FieldEffect/data/size30_M.csv', M, delimiter=',', comments='')

    # data set: magnetic particle core size = 35 nm
    dcore = 35e-9
    dhyd = dcore + 10e-9
    fieldAml_list = 1e-3*np.array([20, 15, 10, 5])
    params.set_params(dCore=dcore, dHyd=dhyd)
    print(params.get_params(), end='\n')
    M = np.zeros( (len(fieldAml_list), len(params.tu)) )
    N = np.zeros( ( len(fieldAml_list), len(params.tu)))
    for i, d in enumerate(fieldAml_list):
        params.set_params(fieldB=d)
        init = params.get_params()
        M[i,:], N[i, :] = CombinedNeelBrown(init_data = init)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(fieldAml_list)-1))
    print()
    np.savetxt('FieldEffect/data/size35_M.csv', M, delimiter=',', comments='')
