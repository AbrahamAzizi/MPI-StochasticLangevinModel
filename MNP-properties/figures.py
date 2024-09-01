from init import *
from utlis import low_pass_filter, peaksInit, peaks_analysis
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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

def colorMap(lenlist, name, colors):
    cmap = LinearSegmentedColormap.from_list(name, colors, N=len(lenlist))
    color_list = [cmap(i / (len(lenlist) - 1)) for i in range(len(lenlist))]
    trsp_list = np.linspace(0.5, 1, len(lenlist))
    return color_list, trsp_list


if __name__ == '__main__':
    params = Params()

    freq = params.f_excitation
    # d_core_list = np.array([20, 30, 40, 50, 60])
    # color_list, trsp_list = colorMap(d_core_list, 'forest', ['lime', 'seagreen', 'forestgreen'] )
    # d_hyd_list = d_core_list + 10
    # Ms = 3e5
    # t = []
    # He = []
    # for i, d in enumerate(zip(d_core_list, d_hyd_list)):
    #     params.set_params(dCore = d[0]*1e-9, dHyd = d[1]*1e-9)
    #     init = params.get_params()
    #     t.append(params.tu / freq)
    #     He.append(params.fieldB * np.cos(2 * np.pi * params.tu))
    #
    # # Magnetization in time for core size
    # with open('effectOfDcore.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     M = [list(map(float, row)) for row in reader]
    # M_lowPass=[]
    # for i in range(len(M)):
    #     M_lowPass.append(low_pass_filter(np.array(M[i]), 1, 10000*freq, 3*freq))
    #
    # fig, ax1 = initialize_figure()
    # for i in range(len(M_lowPass)):
    #     ax1.plot(t[i] * 1e3, He[i] * 1e3, 'black', linewidth=3.0)
    # ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    # ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    # ax1.xaxis.set_tick_params(labelsize=20)
    # ax1.yaxis.set_tick_params(labelsize=20)
    # ax1.plot([], [], 'black', linewidth=3.0, label='$\mu_0 H$ (mT)')
    # set_spines_grid(ax1)
    # ax2 = ax1.twinx()
    # ax2.xaxis.set_tick_params(labelsize=20)
    # ax2.yaxis.set_tick_params(labelsize=20)
    # for i in range(len(M_lowPass)):
    #     ax2.plot(t[i] * 1e3, Ms*M_lowPass[i] * 1e-3 ,linewidth=3.0, color=color_list[i], alpha=trsp_list[i], label=f'$D_c$= {d_core_list[i]} nm')
    # ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    # ax2.set_xlim(.02, .12)
    # set_spines_grid(ax2)
    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    # set_legend_properties(legend)
    # plt.tight_layout()
    # plt.savefig('core-magnetization-time.png')
    #
    # # Magnetization curve for core size
    # fig, ax = initialize_figure(figsize=(12,6))
    # for i in range(len(M)):
    #     l = len(M[i])
    #     k = int(l / params.nPeriod)
    #     ax.plot(He[i][-k:]* 1e3, Ms*M_lowPass[i][-k:] * 1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    # ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    # ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    # set_spines_grid(ax)
    # plt.tight_layout()
    # plt.savefig('core-magnetization-curve.png')
    #
    # # PSF for core size
    # dM=[]
    # for m in M_lowPass:
    #     dM.append(np.diff(np.append(Ms*m,0)))
    # dH=[]
    # for he in He:
    #     dH.append(np.diff(np.append(he,0)))
    #
    # dM = [np.diff(np.append(m,0)) for m in M_lowPass]
    # dH = [np.diff(np.append(he, 0)) for he in He]
    # fig, ax = initialize_figure(figsize=(12,6))
    # for i in range(len(M)):
    #     l = len(M[i])
    #     k = int(l / params.nPeriod)
    #     ax.plot(He[i][-k:]*1e3, dM[i][-k:]/dH[i][-k:] , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    # ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    # ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    # ax.set_xlim(-18,18)
    # ax.set_ylim(0, 300)
    # set_spines_grid(ax)
    # plt.tight_layout()
    # plt.savefig('core-psf.png')
    #
    # # peak and fwhm for core size
    # j = 2 # the magnetization for core size number j
    # Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[j], dM[j], dH[j], params.nPeriod)
    # resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
    # resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
    #
    # fig, ax = initialize_figure(figsize=(12, 6))
    # ax.plot(Hlmask * 1e3, dMdHlmask, linewidth=3.0)
    # ax.plot(resultl['He_peak'] * 1e3, resultl['dmdH_peak'], 'rp', markersize=10, linewidth=3.0)
    # ax.axvspan(resultl['fwhm_left'] * 1e3, resultl['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)
    #
    # ax.plot(Hrmask * 1e3, dMdHrmask, linewidth=3.0)
    # ax.plot(resultr['He_peak'] * 1e3, resultr['dmdH_peak'], 'rp', markersize=10, linewidth=3.0)
    # ax.axvspan(resultr['fwhm_left'] * 1e3, resultr['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)
    #
    # ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    # ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    # ax.set_xlim(-18,18)
    # ax.set_ylim(0, 250)
    # set_spines_grid(ax)
    # plt.tight_layout()
    # plt.savefig('core-fwhm.png')
    #
    # # fwhm and peaks data for core size
    # all_results = []
    # for i in range(len(M)):
    #     Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[i], dM[i], dH[i], params.nPeriod)
    #     resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
    #     resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
    #     combined_result = {'CoreSize': i}  # Include the index as a core list identifier
    #     combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
    #     combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
    #     all_results.append(combined_result)
    #
    # with open('fwhm_peaks_Core.csv', 'w', newline='') as csvfile:
    #     fieldnames = all_results[0].keys()
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for result in all_results:
    #         writer.writerow(result)
    #
    # #Magnetization in time for hyd size
    # d_core = 30e-9
    # params.set_params(dCore=d_core)
    # d_hyd_list = np.array([30, 35, 40, 45, 55, 60])
    # color_list, trsp_list = colorMap(d_hyd_list, 'sunset', ['gold', 'orange', 'darkorange'])
    # Ms = 3e5
    # t = []
    # He = []
    # for i, d in enumerate(d_hyd_list):
    #     params.set_params(dHyd=d * 1e-9)
    #     init = params.get_params()
    #     t.append(params.tu / freq)
    #     He.append(params.fieldB * np.cos(2 * np.pi * params.tu))
    #
    # # Magnetization in time for hyd size
    # with open('effectOfDhydrodynamic.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     M = [list(map(float, row)) for row in reader]
    # M_lowPass = []
    # for i in range(len(M)):
    #     M_lowPass.append(low_pass_filter(np.array(M[i]), 1, 10000 * freq, 3 * freq))
    #
    # fig, ax1 = initialize_figure()
    # for i in range(len(M_lowPass)):
    #     ax1.plot(t[i] * 1e3, He[i] * 1e3, 'black', linewidth=3.0)
    # ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    # ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    # ax1.xaxis.set_tick_params(labelsize=20)
    # ax1.yaxis.set_tick_params(labelsize=20)
    # ax1.plot([], [], 'black', linewidth=3.0, label='$\mu_0 H$ (mT)')
    # set_spines_grid(ax1)
    # ax2 = ax1.twinx()
    # ax2.xaxis.set_tick_params(labelsize=20)
    # ax2.yaxis.set_tick_params(labelsize=20)
    # for i in range(len(M_lowPass)):
    #     ax2.plot(t[i] * 1e3, Ms*M_lowPass[i] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=f'$D_h$= {d_hyd_list[i]} nm')
    # ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    # ax2.set_xlim(.02, .12)
    # set_spines_grid(ax2)
    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    # set_legend_properties(legend)
    # plt.tight_layout()
    # plt.savefig('hyd-magnetization-time.png')
    #
    # # Magnetization curve for hyd size
    # fig, ax = initialize_figure(figsize=(12, 6))
    # for i in range(len(M)):
    #     l = len(M[i])
    #     k = int(l / params.nPeriod)
    #     ax.plot(He[i][-k:] * 1e3, Ms*M_lowPass[i][-k:] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    # ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    # ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    # set_spines_grid(ax)
    # plt.tight_layout()
    # plt.savefig('hyd-magnetization-curve.png')
    #
    # # PSF for hyd size
    # dM = []
    # for m in M_lowPass:
    #     dM.append(np.diff(np.append(Ms*m, 0)))
    # dH = []
    # for he in He:
    #     dH.append(np.diff(np.append(he, 0)))
    #
    # dM = [np.diff(np.append(m, 0)) for m in M_lowPass]
    # dH = [np.diff(np.append(he, 0)) for he in He]
    # fig, ax = initialize_figure(figsize=(12, 6))
    # for i in range(len(M)):
    #     l = len(M[i])
    #     k = int(l / params.nPeriod)
    #     ax.plot(He[i][-k:] * 1e3, dM[i][-k:] / dH[i][-k:], color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    # ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    # ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    # ax.set_xlim(-18, 18)
    # ax.set_ylim(0, 250)
    # set_spines_grid(ax)
    # plt.tight_layout()
    # plt.savefig('hyd-psf.png')
    #
    # # peak and fwhm for hyd size
    # j = 2  # the magnetization for core size number j
    # Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[j], dM[j], dH[j], params.nPeriod)
    # resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
    # resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
    #
    # fig, ax = initialize_figure(figsize=(12, 6))
    # ax.plot(Hlmask * 1e3, dMdHlmask, linewidth=3.0)
    # ax.plot(resultl['He_peak'] * 1e3, resultl['dmdH_peak'], 'rp', markersize=10, linewidth=3.0)
    # ax.axvspan(resultl['fwhm_left'] * 1e3, resultl['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)
    #
    # ax.plot(Hrmask * 1e3, dMdHrmask, linewidth=3.0)
    # ax.plot(resultr['He_peak'] * 1e3, resultr['dmdH_peak'], 'rp', markersize=10, linewidth=3.0)
    # ax.axvspan(resultr['fwhm_left'] * 1e3, resultr['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)
    #
    # ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    # ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    # ax.set_xlim(-18, 18)
    # ax.set_ylim(0, 250)
    # set_spines_grid(ax)
    # plt.tight_layout()
    # plt.savefig('hyd-fwhm.png')
    #
    # # fwhm and peaks data for hyd size
    # all_results = []
    # for i in range(len(M)):
    #      Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[i], dM[i], dH[i], params.nPeriod)
    #      resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
    #      resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
    #      combined_result = {'HydSize': i}  # Include the index as a core list identifier
    #      combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
    #      combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
    #      all_results.append(combined_result)
    #
    # with open('fwhm_peaks_hyd.csv', 'w', newline='') as csvfile:
    #      fieldnames = all_results[0].keys()
    #      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #      writer.writeheader()
    #      for result in all_results:
    #          writer.writerow(result)
    #
    # # Anisotropy
    # d_core = 30e-9
    # d_hyd = 40e-9
    # params.set_params(dCore = d_core, dHyd = d_hyd)
    # ka_list = np.array([3, 5, 8, 10, 15])
    # Ms = 3e5
    # color_list, trsp_list = colorMap(ka_list, 'grapes', ['violet', 'fuchsia', 'mediumvioletred'])
    # t = []
    # He = []
    # for i, d in enumerate(ka_list):
    #     params.set_params(kAnis=d * 1e3)
    #     init = params.get_params()
    #     t.append(params.tu / freq)
    #     He.append(params.fieldB * np.cos(2 * np.pi * params.tu))
    #
    # # Magnetization in time for anisotropy
    # with open('effectOfAnisotropy.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     M = [list(map(float, row)) for row in reader]
    # M_lowPass = []
    # for i in range(len(M)):
    #     M_lowPass.append(low_pass_filter(np.array(M[i]), 1, 10000 * freq, 3 * freq))
    #
    # fig, ax1 = initialize_figure()
    # for i in range(len(M_lowPass)):
    #     ax1.plot(t[i] * 1e3, He[i] * 1e3, 'black', linewidth=3.0)
    # ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    # ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    # ax1.xaxis.set_tick_params(labelsize=20)
    # ax1.yaxis.set_tick_params(labelsize=20)
    # ax1.plot([], [], 'black', linewidth=3.0, label='$\mu_0 H$ (mT)')
    # set_spines_grid(ax1)
    # ax2 = ax1.twinx()
    # ax2.xaxis.set_tick_params(labelsize=20)
    # ax2.yaxis.set_tick_params(labelsize=20)
    # for i in range(len(M_lowPass)):
    #     ax2.plot(t[i] * 1e3, Ms*M_lowPass[i] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=f'$Ka$= {ka_list[i]} kJ/$m^3$')
    # ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    # ax2.set_xlim(.02, .12)
    # set_spines_grid(ax2)
    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    # set_legend_properties(legend)
    # plt.tight_layout()
    # plt.savefig('anis-magnetization-time.png')
    #
    # # Magnetization curve for anisotoropy
    # fig, ax = initialize_figure(figsize=(12, 6))
    # for i in range(len(M)):
    #     l = len(M[i])
    #     k = int(l / params.nPeriod)
    #     ax.plot(He[i][-2*k:-k+100] * 1e3, Ms*M_lowPass[i][-2*k:-k+100] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    # ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    # ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    # set_spines_grid(ax)
    # plt.tight_layout()
    # plt.savefig('anis-magnetization-curve.png')
    #
    # # PSF for anisotropy
    # dM = []
    # for m in M_lowPass:
    #     dM.append(np.diff(np.append(Ms*m, 0)))
    # dH = []
    # for he in He:
    #     dH.append(np.diff(np.append(he, 0)))
    #
    # dM = [np.diff(np.append(m, 0)) for m in M_lowPass]
    # dH = [np.diff(np.append(he, 0)) for he in He]
    # fig, ax = initialize_figure(figsize=(12, 6))
    # for i in range(len(M)):
    #     l = len(M[i])
    #     k = int(l / params.nPeriod)
    #     ax.plot(He[i][-2*k:-k+100] * 1e3, dM[i][-2*k:-k+100] / dH[i][-2*k:-k+100], color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    # ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    # ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    # ax.set_xlim(-18, 18)
    # ax.set_ylim(0, 250)
    # set_spines_grid(ax)
    # plt.tight_layout()
    # plt.savefig('anis-psf.png')
    #
    # # peak and fwhm for anisotropy
    # j = 2  # the magnetization for core size number j
    # Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[j], dM[j], dH[j], params.nPeriod)
    # resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
    # resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
    #
    # fig, ax = initialize_figure(figsize=(12, 6))
    # ax.plot(Hlmask * 1e3, dMdHlmask, linewidth=3.0)
    # ax.plot(resultl['He_peak'] * 1e3, resultl['dmdH_peak'], 'rp', markersize=10, linewidth=3.0)
    # ax.axvspan(resultl['fwhm_left'] * 1e3, resultl['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)
    #
    # ax.plot(Hrmask * 1e3, dMdHrmask, linewidth=3.0)
    # ax.plot(resultr['He_peak'] * 1e3, resultr['dmdH_peak'], 'rp', markersize=10, linewidth=3.0)
    # ax.axvspan(resultr['fwhm_left'] * 1e3, resultr['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)
    #
    # ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    # ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    # ax.set_xlim(-18, 18)
    # ax.set_ylim(0, 250)
    # set_spines_grid(ax)
    # plt.tight_layout()
    # plt.savefig('anis-fwhm.png')
    #
    # # fwhm and peaks data for anisotropy
    # all_results = []
    # for i in range(len(M)-1):
    #     Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[i], dM[i], dH[i], params.nPeriod)
    #     resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
    #     resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
    #     combined_result = {'Anis': i}  # Include the index as a core list identifier
    #     combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
    #     combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
    #     all_results.append(combined_result)
    #
    # with open('fwhm_peaks_anis.csv', 'w', newline='') as csvfile:
    #     fieldnames = all_results[0].keys()
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for result in all_results:
    #         writer.writerow(result)

    # Magnetization Saturation
    d_core = 30e-9
    d_hyd = 40e-9
    ka = 3e3
    params.set_params(dCore = d_core, dHyd = d_hyd, kAnis = ka)
    Ms_list = np.array([50, 100, 200, 300, 400, 500])
    color_list, trsp_list = colorMap(Ms_list, 'red-brown', ['lightcoral', 'red', 'firebrick'] )
    t = []
    He = []
    for i, d in enumerate(Ms_list):
        params.set_params(Ms=d * 1e3)
        init = params.get_params()
        t.append(params.tu / freq)
        He.append(params.fieldB * np.cos(2 * np.pi * params.tu))

    # Magnetization in time for magnetization saturation
    with open('effectOfMagnetization.csv', 'r') as f:
        reader = csv.reader(f)
        M = [list(map(float, row)) for row in reader]
    M_lowPass = []
    for i in range(len(M)):
        M_lowPass.append(low_pass_filter(np.array(M[i]), 1, 10000 * freq, 3 * freq))

    fig, ax1 = initialize_figure()
    for i in range(len(M_lowPass)):
        ax1.plot(t[i] * 1e3, He[i] * 1e3, 'black', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.plot([], [], 'black', linewidth=3.0, label='$\mu_0 H$ (mT)')
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    for i in range(len(M_lowPass)):
        ax2.plot(t[i] * 1e3, M_lowPass[i], color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=f'$Ms$= {Ms_list[i]} kA/m')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.set_xlim(.02, .12)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('satu-magnetization-time1.png')

    # Magnetization curve for magnetization saturation
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(len(M)):
        l = len(M[i])
        k = int(l / params.nPeriod)
        ax.plot(He[i][-2*k:-k+100] * 1e3, Ms_list[i]*M_lowPass[i][-2*k:-k+100], color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('satu-magnetization-curve1.png')

    # PSF for magnetization saturation
    dM = []
    for i, m in enumerate(M_lowPass):
        dM.append(np.diff(np.append(1e3*Ms_list[i]*m, 0)))
    dH = []
    for he in He:
        dH.append(np.diff(np.append(he, 0)))

    dM = [np.diff(np.append(m, 0)) for m in M_lowPass]
    dH = [np.diff(np.append(he, 0)) for he in He]
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(len(M)):
        l = len(M[i])
        k = int(l / params.nPeriod)
        ax.plot(He[i][-2*k:-k+100] * 1e3, dM[i][-2*k:-k+100] / dH[i][-2*k:-k+100], color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(-18, 18)
    ax.set_ylim(0, 250)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('satu-psf1.png')

    # peak and fwhm for magnetization saturation
    j = 2  # the magnetization for magnetization saturation number j
    Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[j], dM[j], dH[j], params.nPeriod)
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
    ax.set_xlim(-18, 18)
    ax.set_ylim(0, 250)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('satu-fwhm1.png')

    # fwhm and peaks data for magnetization saturation
    all_results = []
    for i in range(1, len(M)):
        Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[i], dM[i], dH[i], params.nPeriod)
        resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
        resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
        combined_result = {'Saturation': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('fwhm_peaks_saturation.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)