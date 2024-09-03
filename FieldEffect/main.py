import numpy as np
from init import *
from utlis import *

if __name__ == '__main__':

    params = Params()  # create a parameter instance

    # first data set: magnetic particle core size = 15 nm
    dcore = 1e-9*15
    dhyd = dcore + 10e-9
    fieldAml_list = 1e-3*np.array([20, 15, 10, 5, 2, 1])
    params.set_params(dCore=dcore, dHyd=dhyd)
    print(params.get_params(), end='\n')
    M = np.zeros( (len(fieldAml_list), len(params.tu)) )
    for i, d in enumerate(fieldAml_list):
        params.set_params(fieldB=d)
        init = params.get_params()
        M[i,:], _ = CombinedNeelBrown(init_data = init)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(fieldAml_list)-1))
    print()
    np.savetxt('size15.csv', M, delimiter=',', comments='')

    # first data set: magnetic particle core size = 20 nm
    dcore = 1e-9*20
    dhyd = dcore + 10e-9
    fieldAml_list = 1e-3*np.array([20, 15, 10, 5, 2, 1])
    params.set_params(dCore=dcore, dHyd=dhyd)
    print(params.get_params(), end='\n')
    M = np.zeros( (len(fieldAml_list), len(params.tu)) )
    for i, d in enumerate(fieldAml_list):
        params.set_params(fieldB=d)
        init = params.get_params()
        M[i,:], _ = CombinedNeelBrown(init_data = init)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(fieldAml_list)-1))
    print()
    np.savetxt('size20.csv', M, delimiter=',', comments='')

    # first data set: magnetic particle core size = 30 nm
    dcore = 1e-9*30
    dhyd = dcore + 10e-9
    fieldAml_list = 1e-3*np.array([20, 15, 10, 5, 2, 1])
    params.set_params(dCore=dcore, dHyd=dhyd)
    print(params.get_params(), end='\n')
    M = np.zeros( (len(fieldAml_list), len(params.tu)) )
    for i, d in enumerate(fieldAml_list):
        params.set_params(fieldB=d)
        init = params.get_params()
        M[i,:], _ = CombinedNeelBrown(init_data = init)
        print('\r', 'fieldAmplitued_list: ' + "." * 10 + " ", end=str(i)+'/'+str(len(fieldAml_list)-1))
    print()
    np.savetxt('size30.csv', M, delimiter=',', comments='')