from init import *
from utlis import low_pass_filter, moving_average_filter, peaksInit, peaks_analysis, Ht
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
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
    params = Params( )

    f = params.f_excitation
    t = params.tu / f
    Ms = 3e5

    # for induced voltages and harmonics
    u0 = 4 * np.pi * 1e-7  # T.m/A, v.s/A/m
    gz = 3  # T/m
    pz = 20e-3 * 795.7747 / 1.59  # A/m/A   1T = 795.7747 A/m, I = 1.59 A
    lz = params.fieldB / gz
    dift = np.diff(np.append(t, 0))
    dHz_free = (lz ** 3) * np.diff(np.append(Ht(f, t), 0))
    uz_free = -pz * u0 * (dHz_free / dift)
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    N = len(freqs) // 2
    x = np.fft.fftshift(freqs / f)[N:]

    # Magnetization in time for core size 20 nm
    M = genfromtxt('FieldEffect/size25.csv', delimiter=',')
    Mfilt=np.zeros(M.shape)
    for i in range(M.shape[0]):
        filtSignal = low_pass_filter(M[i,:], 4, 10000*f, 10*f)
        Mfilt[i,:] = moving_average_filter(filtSignal, 500)

    n, l = M.shape
    k = int(l / params.nPeriod)
    fieldAml_list = np.array([20, 15, 10, 5])
    He = np.zeros(M.shape)
    for i, a in enumerate(fieldAml_list):
        He[i,:] = np.array(a*1e-3 * np.cos(2 * np.pi * params.tu))

    # these values are adjusted to avoid the edge effect in field derivatives
    fieldRange = np.array([[-19.9e-3, 19.9e-3],
                  [-14.9e-3, 14.9e-3],
                  [-9.9e-3, 9.9e-3],
                  [-4.9e-3, 4.9e-3]])
    mask = []
    mask.append(np.where((He[0, -2*k:-k] >= fieldRange[0,0]) & (He[0, -2*k:-k] <= fieldRange[0,1]))[0])
    mask.append(np.where((He[1, -2*k:-k] >= fieldRange[1,0]) & (He[1, -2*k:-k] <= fieldRange[1,1]))[0])
    mask.append(np.where((He[2, -2*k:-k] >= fieldRange[2,0]) & (He[2, -2*k:-k] <= fieldRange[2,1]))[0])
    mask.append(np.where((He[3, -2*k:-k] >= fieldRange[3,0]) & (He[3, -2*k:-k] <= fieldRange[3,1]))[0])

    color_list, trsp_list = colorMap(fieldAml_list, 'forest', ['lime', 'seagreen', 'forestgreen'])
    fig, ax1 = initialize_figure()
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.set_xlim(.01, .11)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    for i in range(M.shape[0]):
        ax1.plot(t[0::100] * 1e3, He[i][0::100] * 1e3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$\mu_0H$= {fieldAml_list[i]} mT')
        ax2.plot(t[0::100] * 1e3, Ms*Mfilt[i, :][0::100] * 1e-3 , '--', color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.set_xlim(.01, .11)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('FieldEffect/core25-magnetization-time.png')

    # Magnetization curve for core size 20nm
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[i, -2*k:-k]*1e3, Ms*Mfilt[i, -2*k:-k]*1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/core25-magnetization-curve.png')

    # Harmonics for core size 20
    fig, ax = initialize_figure(figsize=(16,6))
    for i in range(M.shape[0]):
        dHz = (lz ** 3) * np.diff(np.append(u0 * Ms * Mfilt[i, :], 0))
        uz = -pz * u0 * (dHz / dift)
        unet = uz - uz_free
        uk = np.fft.fft(unet)
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        # Filter x and y for integer values of x from 1 to 20
        x_int = np.array([2*k+1 for k in range(1, 21)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        #markerline, stemlines, baseline = ax.stem(x_int, y_int, bottom=0, markerfmt="Dr")
        ax.plot(x_int, y_int, color=color_list[i], marker='D', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(2, 20)
    ax.set_xticks(range(3, 21, 2))  # Set x-ticks every 2 units
    ax.set_ylabel(r'Harmonics Magnitude($\mu$v)', weight='bold', fontsize=20)
    ax.set_xlabel('f/fe', weight='bold', fontsize=20)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/core25-harmonics.png')

    # psf for core size 25
    for i in range(M.shape[0]):
        Mfilt[i,:] = low_pass_filter(M[i,:], 4, 10000*f, 3*f)
    dM = np.diff(np.append(Mfilt, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, np.zeros((He.shape[0], 1)), axis=1), axis=1)
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]-1):
        ax.plot(He[i, -2*k:-k][mask[i]]*1e3, dM[i, -2*k:-k][mask[i]]/dH[i, -2*k:-k][mask[i]] , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_ylim(10, 300)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/core25-psf.png')

    # fwhm and peaks for core size 20nm
    j = 0 # the field with 5 mT amplitude, the H_range needs to be adjusted based on the mask defined above
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

    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(fieldRange[j,0]*1e3,fieldRange[j,1]*1e3)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/core25-fwhm.png')

    # fwhm and peaks data for core size 25
    all_results = []
    for i in range(M.shape[0]-1):  # not for the last field amplitude without sigmoidal M-H curve
        Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[i], dM[i], dH[i], params.nPeriod,
                                                                       H_range=(fieldRange[i, 0],fieldRange[i,1]))
        resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
        resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
        combined_result = {'CoreSize': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('FieldEffect/fwhm_peaks_Core20.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

   # Magnetization in time for core size 30 nm
    M = genfromtxt('FieldEffect/size30.csv', delimiter=',')
    Mfilt=np.zeros(M.shape)
    for i in range(M.shape[0]):
        filtSignal = low_pass_filter(M[i,:], 4, 10000*f, 10*f)
        Mfilt[i,:] = moving_average_filter(filtSignal, 500)

    color_list, trsp_list = colorMap(fieldAml_list, 'red-brown', ['lightcoral', 'red', 'firebrick'] )
    fig, ax1 = initialize_figure()
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.set_xlim(.01, .11)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    for i in range(M.shape[0]):
        ax1.plot(t[0::100] * 1e3, He[i][0::100] * 1e3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$\mu_0H$= {fieldAml_list[i]} mT')
        ax2.plot(t[0::100] * 1e3, Ms * Mfilt[i, :][0::100] * 1e-3, '--', color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.set_xlim(.01, .11)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('FieldEffect/core30-magnetization-time.png')

    # Magnetization curve for core size 30nm
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        ax.plot(He[i, -2*k:-k] * 1e3, Ms *  Mfilt[i, -2*k:-k] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/core30-magnetization-curve.png')

    # Harmonics for core size 30
    fig, ax = initialize_figure(figsize=(16,6))
    for i in range(M.shape[0]):
        dHz = (lz ** 3) * np.diff(np.append(u0 * Ms * Mfilt[i, :], 0))
        uz = -pz * u0 * (dHz / dift)
        unet = uz - uz_free
        uk = np.fft.fft(unet)
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        # Filter x and y for integer values of x from 1 to 20
        x_int = np.array([2*k+1 for k in range(1, 21)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        #markerline, stemlines, baseline = ax.stem(x_int, y_int, bottom=0, markerfmt="Dr")
        ax.plot(x_int, y_int, color=color_list[i], marker='D', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(2, 20)
    ax.set_xticks(range(3, 21, 2))  # Set x-ticks every 2 units
    ax.set_ylabel(r'Harmonics Magnitude($\mu$v)', weight='bold', fontsize=20)
    ax.set_xlabel('f/fe', weight='bold', fontsize=20)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/Core30-harmonics.png')

    # psf for core size 30nm
    for i in range(M.shape[0]):
        Mfilt[i,:] = low_pass_filter(M[i,:], 4, 10000*f, 3*f)
    
    dM = np.diff(np.append(Mfilt, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, np.zeros((He.shape[0], 1)), axis=1), axis=1)
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]-1):
        ax.plot(He[i, -2*k:-k][mask[i]] * 1e3, dM[i, -2*k:-k][mask[i]] / dH[i, -2*k:-k][mask[i]], color=color_list[i], alpha=trsp_list[i], linewidth=3.0)

    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_ylim(10, 300)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/core30-psf.png')

    #fwhm and peaks for core size 30nm
    j = 0 # the field with 5 mT amplitude, the H_range needs to be adjusted based on the fieldRange
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

    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(fieldRange[j,0]*1e3,fieldRange[j,1]*1e3)
    ax.set_ylim(0, 300)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/core30-fwhm.png')

    # fwhm and peaks data for core size 30
    all_results = []
    for i in range(M.shape[0]-1):  # not for the last two field amplitudes
        Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[i], dM[i], dH[i], params.nPeriod,
                                                                       H_range=(fieldRange[i, 0],fieldRange[i,1]))
        resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
        resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
        combined_result = {'CoreSize': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('FieldEffect/fwhm_peaks_Core30.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)


    # Magnetization in time for core size 35 nm
    M = genfromtxt('FieldEffect/size35.csv', delimiter=',')
    Mfilt=np.zeros(M.shape)
    for i in range(M.shape[0]):
        filtSignal = low_pass_filter(M[i,:], 4, 10000*f, 10*f)
        Mfilt[i,:] = moving_average_filter(filtSignal, 500)

    color_list, trsp_list = colorMap(fieldAml_list, 'sunset', ['gold', 'orange', 'darkorange'])
    fig, ax1 = initialize_figure()
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.set_xlim(.01, .11)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    for i in range(M.shape[0]):
        ax1.plot(t[0::100] * 1e3, He[i][0::100] * 1e3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$\mu_0H$= {fieldAml_list[i]} mT')
        ax2.plot(t[0::100] * 1e3, Ms * Mfilt[i, :][0::100] * 1e-3, '--', color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.set_xlim(.01, .11)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('FieldEffect/core35-magnetization-time.png')

    # Magnetization curve for core size 35nm
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        ax.plot(He[i, -2*k:-k] * 1e3, Ms *  Mfilt[i, -2*k:-k] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/core35-magnetization-curve.png')

    # Harmonics for core size 35
    fig, ax = initialize_figure(figsize=(16,6))
    for i in range(M.shape[0]):
        dHz = (lz ** 3) * np.diff(np.append(u0 * Ms * Mfilt[i, :], 0))
        uz = -pz * u0 * (dHz / dift)
        unet = uz - uz_free
        uk = np.fft.fft(unet)
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        # Filter x and y for integer values of x from 1 to 20
        x_int = np.array([2*k+1 for k in range(1, 21)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        #markerline, stemlines, baseline = ax.stem(x_int, y_int, bottom=0, markerfmt="Dr")
        ax.plot(x_int, y_int, color=color_list[i], marker='D', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(2, 20)
    ax.set_xticks(range(3, 21, 2))  # Set x-ticks every 2 units
    ax.set_ylabel(r'Harmonics Magnitude($\mu$v)', weight='bold', fontsize=20)
    ax.set_xlabel('f/fe', weight='bold', fontsize=20)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/Core35-harmonics.png')

    # psf for core size 35nm
    for i in range(M.shape[0]):
        Mfilt[i,:] = low_pass_filter(M[i,:], 4, 10000*f, 3*f)
    
    dM = np.diff(np.append(Mfilt, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, np.zeros((He.shape[0], 1)), axis=1), axis=1)
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]-1):
        ax.plot(He[i, -2*k:-k][mask[i]] * 1e3, dM[i, -2*k:-k][mask[i]] / dH[i, -2*k:-k][mask[i]], color=color_list[i], alpha=trsp_list[i], linewidth=3.0)

    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_ylim(10, 350)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/core35-psf.png')

    # fwhm and peaks for core size 35nm
    j = 0 # the field with 5 mT amplitude, the H_range needs to be adjusted based on the fieldRange
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

    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(fieldRange[j,0]*1e3,fieldRange[j,1]*1e3)
    ax.set_ylim(0, 300)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/core35-fwhm.png')

    # fwhm and peaks data for core size 40
    all_results = []
    for i in range(M.shape[0]-1):  # not for the last two field amplitudes
        Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He[i], dM[i], dH[i], params.nPeriod,
                                                                       H_range=(fieldRange[i, 0],fieldRange[i,1]))
        resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
        resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
        combined_result = {'CoreSize': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('FieldEffect/fwhm_peaks_Core35.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    # Magnetization in time for core size 40 nm
    M = genfromtxt('FieldEffect/size40.csv', delimiter=',')
    Mfilt=np.zeros(M.shape)
    for i in range(M.shape[0]):
        filtSignal = low_pass_filter(M[i,:], 4, 10000*f, 10*f)
        Mfilt[i,:] = moving_average_filter(filtSignal, 500)

    He = np.zeros(M.shape)
    for i, a in enumerate(fieldAml_list):
        He[i, :] = np.array(a * 1e-3 * np.cos(2 * np.pi * params.tu))

    color_list, trsp_list = colorMap(fieldAml_list, 'grapes', ['violet', 'fuchsia', 'mediumvioletred'])
    fig, ax1 = initialize_figure()
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.set_xlim(.01, .11)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    for i in range(M.shape[0]):
        ax1.plot(t[0::100] * 1e3, He[i][0::100] * 1e3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$\mu_0H$= {fieldAml_list[i]} mT')
        ax2.plot(t[0::100] * 1e3, Ms *Mfilt[i, :][0::100] * 1e-3, '--', color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.set_xlim(.01, .11)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('FieldEffect/core40-magnetization-time.png')

    # Magnetization curve for core size 40nm
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        ax.plot(He[i, -2*k:-k] * 1e3, Ms * Mfilt[i, -2*k:-k] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/core40-magnetization-curve.png')

    # Harmonics for core size 40
    fig, ax = initialize_figure(figsize=(16,6))
    for i in range(M.shape[0]):
        dHz = (lz ** 3) * np.diff(np.append(u0 * Ms * Mfilt[i, :], 0))
        uz = -pz * u0 * (dHz / dift)
        unet = uz - uz_free
        uk = np.fft.fft(unet)
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        # Filter x and y for integer values of x from 1 to 20
        x_int = np.array([2*k+1 for k in range(1, 21)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        #markerline, stemlines, baseline = ax.stem(x_int, y_int, bottom=0, markerfmt="Dr")
        ax.plot(x_int, y_int, color=color_list[i], marker='D', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(2, 20)
    ax.set_xticks(range(3, 21, 2))  # Set x-ticks every 2 units
    ax.set_ylabel(r'Harmonics Magnitude($\mu$v)', weight='bold', fontsize=20)
    ax.set_xlabel('f/fe', weight='bold', fontsize=20)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/Core40-harmonics.png')

    # PSF for core size 40nm
    for i in range(M.shape[0]):
        Mfilt[i,:] = low_pass_filter(M[i,:], 4, 10000*f, 3*f)
    dM = np.diff(np.append(Mfilt, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, np.zeros((He.shape[0], 1)), axis=1), axis=1)
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]-1):
        ax.plot(He[i, -2*k:-k][mask[i]]*1e3, dM[i, -2*k:-k][mask[i]]/dH[i, -2*k:-k][mask[i]] , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)

    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_ylim(10, 400)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/core40-psf.png')

    # fwhm and peaks for core size 40
    j = 0 # the magnetization for core size number j and it's corrspond H
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

    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(fieldRange[j,0]*1e3,fieldRange[j,1]*1e3)
    ax.set_ylim(0, 300)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('FieldEffect/core40-fwhm.png')

    # fwhm and peaks data for core size 40
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

    with open('FieldEffect/fwhm_peaks_Core40.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
