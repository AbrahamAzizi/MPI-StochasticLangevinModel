import h5py
from init import *
from utils import MPI_Langevin_std_core, InducedVoltage, psf, fwhm_data

def save_data(group, data, M, N, xif, sigf, psfval, uz, tcv, fwhml, fwhmr):
    data_subgroup = group.create_group('init_data')
    for key, value in vars(data).items():
        data_subgroup.attrs[key] = value 

    for i, d in enumerate(stdCore_list):
        subgroup_name = f"std_{d:.2f}"  
        subgroup = group.create_group(subgroup_name)
        subgroup.create_dataset("Mz", data=M[i, :, 2])
        subgroup.create_dataset("Nz", data=N[i, :, 2])
        subgroup.create_dataset("AC_field", data=xif[i, :, 2])
        subgroup.create_dataset("anisotropy_field", data=sigf[i])
        subgroup.create_dataset("psf", data=psfval[i])
        subgroup.create_dataset("uz", data=uz[i])
        subgroup.attrs["tcv"] = tcv

        leftpeak_subgroup = subgroup.create_group("peak_left")
        for key, value in fwhml[f"std_{d:.2f}"].items():
            leftpeak_subgroup.create_dataset(key, data=value)
        rightpeak_subgroup = subgroup.create_group("peak_right")
        for key, value in fwhmr[f"std_{d:.2f}"].items():
            rightpeak_subgroup.create_dataset(key, data=value)

if __name__ == '__main__':    
    pz = 20e-3 * 795.7747 / 1.59  # A/m/A   1T = 795.7747 A/m, I = 1.59 A
    data = Data() 
    f = data.fieldFreq
    cycs = data.nPeriod
    data.rsol = 100
    rsol = data.rsol
    fs = rsol*2*f
    dt = 1/fs
    tf = cycs*(1/f)
    lent = int(np.ceil(tf/dt))
    stdList = np.array([.5, 1, 3, 5, 7])

    # Prepare for all groups
    data_groups = {
        'IPG30': (29.53*1e-9, 16*1e-9, .09),
        'SHS30': (29.52*1e-9, 28*1e-9, .08),
        'SHP25': (25.65*1e-9, 16*1e-9, .05),
        'SHP15': (12.42*1e-9, 16*1e-9, .11)
    }

    with h5py.File('MNP-sizeDistribution/data/data.h5', 'w') as f:
        for group_name, (meanCore, coatingSize, stdCoreFactor) in data_groups.items():
            # Reset data for each group
            data.meanCore = meanCore
            data.coatingSize = coatingSize 
            stdCore_list = np.round(stdCoreFactor * stdList, 3)
            
            M = np.zeros((len(stdCore_list), lent, 3))
            N = np.zeros((len(stdCore_list), lent, 3))
            xif = np.zeros((len(stdCore_list), lent, 3))
            sigf = np.zeros((len(stdCore_list), lent, 3))
            tcv = np.zeros(len(stdCore_list))
            psfval = np.zeros((len(stdCore_list), lent-1))
            uz = np.zeros((len(stdCore_list), lent-1))
            fwhml = {}
            fwhmr = {}

            # Perform calculations for the current group
            print(f"\n{group_name}: ")
            for i, d in enumerate(stdCore_list):
                print(f"\nsample std={d} ----- start", flush=True)
                data.stdCore = d
                M[i], N[i], xif[i], sigf[i], tcv[i] = MPI_Langevin_std_core(data)
                psfval[i] = psf(M[i, :, 2], xif[i, :, 2], sigf[i, :, 2])
                uz[i] = InducedVoltage(M[i, :, 2], xif[i, :, 2], pz, fs)
                fwhml[f"std_{d:.2f}"], fwhmr[f"std_{d:.2f}"] = fwhm_data(psfval[i], xif[i, :, 2])
                print(f"\nsample std={d} ----- completed.")

            if group_name in f:
                del f[group_name]
            group = f.create_group(group_name)
            save_data(group, data, M, N, xif, sigf, psfval, uz, tcv, fwhml, fwhmr)

            print(f"\n{group_name}_data ----- saved.")
