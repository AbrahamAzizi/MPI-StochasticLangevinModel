from init import *
from utlis import low_pass_filter, peaksInit, peaks_analysis
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import csv

def initialize_figure(figsize=(18,6), dpi=300, font_scale=2):
    sns.set_context("notebook", font_scale=font_scale, )
    sns.set_style("whitegrid")
    rcParams['font.weight'] = 'bold'
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    return fig, ax
def set_spines_grid(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

def set_legend_properties(legend):
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1)


if __name__ == '__main__':
    params = Params( )

    f = params.f_excitation
    t = params.tu / f
    Ms = 3e5

    # Magnetization in time for core size 15 nm
    M = genfromtxt('size15.csv', delimiter=',')
    n, l = M.shape
    k = int(l / params.nPeriod)
    fieldAml_list = np.array([20, 15, 10, 5, 2, 1])
    He = np.zeros(M.shape)
    for i, a in enumerate(fieldAml_list):
        He[i,:] = np.array(a*1e-3 * np.cos(2 * np.pi * params.tu))

    # these values are adjusted to avoid the edge effect in field derivatives
    fieldRange = np.array([[-17e-3, 17e-3],
                  [-12e-3, 12e-3],
                  [-9e-3, 9e-3],
                  [-4.8e-3, 4.8e-3],
                  [-5e-3, 5e-3],
                  [-.8e-3, .8e-3]])
    mask = []
    mask.append(np.where((He[0, -k:] >= fieldRange[0,0]) & (He[0, -k:] <= fieldRange[0,1]))[0])
    mask.append(np.where((He[1, -k:] >= fieldRange[1,0]) & (He[1, -k:] <= fieldRange[1,1]))[0])
    mask.append(np.where((He[2, -k:] >= fieldRange[2,0]) & (He[2, -k:] <= fieldRange[2,1]))[0])
    mask.append(np.where((He[3, -k:] >= fieldRange[3,0]) & (He[3, -k:] <= fieldRange[3,1]))[0])
    mask.append(np.where((He[4, -k:] >= fieldRange[4,0]) & (He[4, -k:] <= fieldRange[4,1]))[0])
    mask.append(np.where((He[5, -k:] >= fieldRange[5,0]) & (He[5, -k:] <= fieldRange[5,1]))[0])

    fig, ax1 = initialize_figure()
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    for i in range(M.shape[0]):
        ax1.plot(t[0::100] * 1e3, He[i][0::100] * 1e3, linewidth=3.0, label=f'$\mu_0H$= {fieldAml_list[i]} mT')
        ax2.plot(t[0::100] * 1e3, Ms*M[i, :][0::100] * 1e-3 , '--', linewidth=3.0)
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('core15-magnetization-time.png')

    # Magnetization curve for core size 15nm
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[i, -k:]* 1e3, Ms*M[i, -k:] * 1e-3 ,linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('core15-magnetization-curve.png')

    # psf for core size 15
    M_lowPass=np.zeros(M.shape)
    for i in range(M.shape[0]):
        M_lowPass[i,:] = low_pass_filter(M[i,:], 1, 1000*f, 3*f)
    dM = np.diff(np.append(M_lowPass, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, np.zeros((He.shape[0], 1)), axis=1), axis=1)
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[i, -k:][mask[i]]*1e3, dM[i, -k:][mask[i]]/dH[i, -k:][mask[i]] ,linewidth=3.0)

    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_ylim(0, 1500)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('core15-psf.png')

    # fwhm and peaks for core size 15nm
    j = 3 # the field with 5 mT amplitude, the H_range needs to be adjusted based on the mask defined above
    Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[j], dM[j], dH[j], params.nPeriod,
                                                                   H_range=(fieldRange[j, 0],fieldRange[j,1]))
    resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
    resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)

    fig, ax = initialize_figure(figsize=(12, 6))
    ax.plot(Hlmask * 1e3, dMdHlmask, linewidth=3.0)
    ax.plot(resultl['He_peak'] * 1e3, resultl['dmdH_peak'], 'rp', markersize=10, linewidth=3.0)
    ax.axvspan(resultl['fwhm_left'] * 1e3, resultl['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)

    ax.plot(Hrmask * 1e3, dMdHrmask, linewidth=3.0)
    ax.plot(resultr['He_peak'] * 1e3, resultr['dmdH_peak'], 'rp', markersize=10, linewidth=3.0)
    ax.axvspan(resultr['fwhm_left'] * 1e3, resultr['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)

    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(fieldRange[j,0]*1e3,fieldRange[j,1]*1e3)
    ax.set_ylim(0, 1500)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('core15-fwhm.png')

    # fwhm and peaks data for core size 15
    all_results = []
    for i in range(M.shape[0]-2):  # not for the last two field amplitudes
        Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[i], dM[i], dH[i], params.nPeriod,
                                                                       H_range=(fieldRange[i, 0],fieldRange[i,1]))
        resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
        resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
        combined_result = {'CoreSize': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('fwhm_peaks_Core15.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    # Magnetization in time for core size 20 nm
    M = genfromtxt('size20.csv', delimiter=',')

    fig, ax1 = initialize_figure()
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    for i in range(M.shape[0]):
        ax1.plot(t[0::100] * 1e3, He[i][0::100] * 1e3, linewidth=3.0, label=f'$\mu_0H$= {fieldAml_list[i]} mT')
        ax2.plot(t[0::100] * 1e3, Ms * M[i, :][0::100] * 1e-3, '--', linewidth=3.0)
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('core20-magnetization-time.png')

    # Magnetization curve for core size 20nm
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        ax.plot(He[i, -k:] * 1e3, Ms * M[i, -k:] * 1e-3, linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('core20-magnetization-curve.png')

    # psf for core size 20nm
    M_lowPass = np.zeros(M.shape)
    mask[3] = np.where((He[3, -k:] >= -4.9e-3) & (He[3, -k:] <= 4.9e-3))[0]
    for i in range(M.shape[0]):
        M_lowPass[i, :] = low_pass_filter(M[i, :], 1, 1000 * f, 3 * f)
    dM = np.diff(np.append(M_lowPass, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, np.zeros((He.shape[0], 1)), axis=1), axis=1)
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        ax.plot(He[i, -k:][mask[i]] * 1e3, dM[i, -k:][mask[i]] / dH[i, -k:][mask[i]], linewidth=3.0)

    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_ylim(0, 1500)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('core30-psf.png')

    # fwhm and peaks for core size 20nm
    j = 3 # the field with 5 mT amplitude, the H_range needs to be adjusted based on the fieldRange
    Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[j], dM[j], dH[j], params.nPeriod,
                                                                   H_range=(fieldRange[j,0],fieldRange[j,1]))
    resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
    resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)

    fig, ax = initialize_figure(figsize=(12, 6))
    ax.plot(Hlmask * 1e3, dMdHlmask, linewidth=3.0)
    ax.plot(resultl['He_peak'] * 1e3, resultl['dmdH_peak'], 'rp', markersize=10, linewidth=3.0)
    ax.axvspan(resultl['fwhm_left'] * 1e3, resultl['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)

    ax.plot(Hrmask * 1e3, dMdHrmask, linewidth=3.0)
    ax.plot(resultr['He_peak'] * 1e3, resultr['dmdH_peak'], 'rp', markersize=10, linewidth=3.0)
    ax.axvspan(resultr['fwhm_left'] * 1e3, resultr['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)

    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(fieldRange[j,0]*1e3,fieldRange[j,1]*1e3)
    ax.set_ylim(0, 1500)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('core20-fwhm.png')

    # fwhm and peaks data for core size 20
    all_results = []
    for i in range(M.shape[0]-2):  # not for the last two field amplitudes
        Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[i], dM[i], dH[i], params.nPeriod,
                                                                       H_range=(fieldRange[i, 0],fieldRange[i,1]))
        resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
        resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
        combined_result = {'CoreSize': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('fwhm_peaks_Core20.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    # Magnetization in time for core size 30 nm
    M = genfromtxt('size30.csv', delimiter=',')
    He = np.zeros(M.shape)
    for i, a in enumerate(fieldAml_list):
        He[i, :] = np.array(a * 1e-3 * np.cos(2 * np.pi * params.tu))

    fig, ax1 = initialize_figure()
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    for i in range(M.shape[0]):
        ax1.plot(t[0::100] * 1e3, He[i][0::100] * 1e3, linewidth=3.0, label=f'$\mu_0H$= {fieldAml_list[i]} mT')
        ax2.plot(t[0::100] * 1e3, Ms * M[i, :][0::100] * 1e-3, '--', linewidth=3.0)
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('core30-magnetization-time.png')

    # Magnetization curve for core size 30nm
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        ax.plot(He[i, -k:] * 1e3, Ms * M[i, -k:] * 1e-3, linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('core30-magnetization-curve.png')

    # PSF for core size 30nm
    M_lowPass = np.zeros(M.shape)
    mask[3] = np.where((He[3, -k:] >= -4.98e-3) & (He[3, -k:] <= 4.98e-3))[0]
    for i in range(M.shape[0]):
        M_lowPass[i, :] = low_pass_filter(M[i, :], 1, 1000 * f, 3 * f)
    dM = np.diff(np.append(M_lowPass, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, np.zeros((He.shape[0], 1)), axis=1), axis=1)
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        ax.plot(He[i, -k:][mask[i]]*1e3, dM[i, -k:][mask[i]]/dH[i, -k:][mask[i]] ,linewidth=3.0)

    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_ylim(0, 1500)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('core30-psf.png')

    # fwhm and peaks for core size 30
    j = 3 # the magnetization for core size number j and it's corrspond H
    Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[j], dM[j], dH[j], params.nPeriod,
                                                                   H_range=(fieldRange[j,0], fieldRange[j,1]))
    resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
    resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)

    fig, ax = initialize_figure(figsize=(12, 6))
    ax.plot(Hlmask * 1e3, dMdHlmask, linewidth=3.0)
    ax.plot(resultl['He_peak'] * 1e3, resultl['dmdH_peak'], 'rp', markersize=10, linewidth=3.0)
    ax.axvspan(resultl['fwhm_left'] * 1e3, resultl['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)

    ax.plot(Hrmask * 1e3, dMdHrmask, linewidth=3.0)
    ax.plot(resultr['He_peak'] * 1e3, resultr['dmdH_peak'], 'rp', markersize=10, linewidth=3.0)
    ax.axvspan(resultr['fwhm_left'] * 1e3, resultr['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)

    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(fieldRange[j,0]*1e3,fieldRange[j,1]*1e3)
    ax.set_ylim(0, 1500)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('core30-fwhm.png')

    # fwhm and peaks data for core size 30
    all_results = []
    for i in range(M.shape[0]-2):
        Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[i], dM[i], dH[i], params.nPeriod,
                                                                       H_range=(fieldRange[i, 0],fieldRange[i,1]))
        resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
        resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
        combined_result = {'CoreSize': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('fwhm_peaks_Core30.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

