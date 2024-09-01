from init import *
from utlis import low_pass_filter, peaksInit, peaks_analysis, Ht
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
import seaborn as sns
import csv

def initialize_figure(figsize=(20,6), dpi=300, font_scale=2):
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

    f = params.f_excitation
    t = params.tu / f
    He = params.fieldB * Ht(f, t)
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

    # IGP30 magnetization-time
    M = genfromtxt('IPG30.csv', delimiter=',')
    M_lowPass=np.zeros(M.shape)
    for i in range(M.shape[0]):
        M_lowPass[i,:] = low_pass_filter(M[i,:], 1, 10000*f, 3*f)
    stdIGP30 = .09
    sigmaCore_list = np.round(stdIGP30*np.array([.5, 1, 3, 5, 10]),3)
    color_list, trsp_list = colorMap(sigmaCore_list, 'forest', ['lime', 'seagreen', 'forestgreen'])
    fig, ax1 = initialize_figure()
    ax1.plot(t * 1e3, He * 1e3, 'black', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    for i in range(M.shape[0]):
        ax2.plot(t * 1e3, Ms*M_lowPass[i, :] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=f'$\sigma$= {sigmaCore_list[i]}')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('IGP30-magnetization-time.png')

    # IPG30 Magnetization curve
    n, l = M.shape
    k = int(l / params.nPeriod)
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[-k:]* 1e3, Ms*M_lowPass[i, -k:] * 1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('IPG30-magnetization-curve.png')

    # IPG30 PSF
    dM = np.diff(np.append(M_lowPass, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, 0))
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[-k:]*1e3, dM[i, -k:]/dH[-k:] , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(-18,18)
    ax.set_ylim(0, 200)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('IPG30-psf.png')

    # IPG30 harmonics
    fig, ax = initialize_figure(figsize=(16,6))
    for i in range(M.shape[0]):
        dHz = (lz ** 3) * np.diff(np.append(u0 * Ms * M_lowPass[i, :], 0))
        uz = -pz * u0 * (dHz / dift)
        unet = uz - uz_free
        uk = np.fft.fft(unet)
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        # Filter x and y for integer values of x from 1 to 20
        x_int = np.array([2*k+1 for k in range(21)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        #markerline, stemlines, baseline = ax.stem(x_int, y_int, bottom=0, markerfmt="Dr")
        ax.plot(x_int, y_int, color=color_list[i], marker='D', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
        # Set color and alpha for stem lines
        #plt.setp(stemlines, color=color_list[i], alpha=trsp_list[i], linewidth=5)
        # Set color and alpha for markers
        #plt.setp(markerline, color=color_list[i], alpha=trsp_list[i], markersize=15)
        # Set color and alpha for baseline (usually invisible)
        #plt.setp(baseline, visible=False)
    ax.set_xlim(0, 20)
    ax.set_xticks(range(1, 21, 2))  # Set x-ticks every 2 units
    ax.set_ylabel('Harmonics Magnitude($\mu$v)', weight='bold', fontsize=20)
    ax.set_xlabel('f/fe', weight='bold', fontsize=20)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('IPG30-harmonics.png')

    # fwhm and peaks data for IPG30
    all_results = []
    for i in range(len(M)):
        Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He, dM[i], dH, params.nPeriod)
        resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
        resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
        combined_result = {'sigma': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('fwhm_IPG30.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    # SHS30 magnetization-time
    M = genfromtxt('SHS30.csv', delimiter=',')
    M_lowPass=np.zeros(M.shape)
    for i in range(M.shape[0]):
        M_lowPass[i,:] = low_pass_filter(M[i,:], 1, 10000*f, 3*f)
    stdSHS30 = .08
    sigmaCore_list = np.round(stdSHS30*np.array([.5, 1, 3, 5, 10]),3)
    color_list, trsp_list = colorMap(sigmaCore_list, 'sunset', ['gold', 'orange', 'darkorange'])
    fig, ax1 = initialize_figure()
    ax1.plot(t * 1e3, He * 1e3, 'black', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    for i in range(M.shape[0]):
        ax2.plot(t * 1e3, Ms*M_lowPass[i, :] * 1e-3 ,linewidth=3.0, color=color_list[i], alpha=trsp_list[i], label=f'$\sigma$= {sigmaCore_list[i]}')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('SHS30-magnetization-time.png')

    # SHS30 Magnetization curve
    n, l = M.shape
    k = int(l / params.nPeriod)
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[-k:]* 1e3, Ms*M_lowPass[i, -k:] * 1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('SHS30-magnetization-curve.png')

    # SHS30 PSF
    dM = np.diff(np.append(M_lowPass, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, 0))
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[-k:]*1e3, dM[i, -k:]/dH[-k:] , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(-18,18)
    ax.set_ylim(0, 200)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('SHS30-psf.png')

    # SHS30 harmonics
    fig, ax = initialize_figure(figsize=(16,6))
    for i in range(M.shape[0]):
        dHz = (lz ** 3) * np.diff(np.append(u0 * Ms * M_lowPass[i, :], 0))
        uz = -pz * u0 * (dHz / dift)
        unet = uz - uz_free
        uk = np.fft.fft(unet)
        y = 1e6 * abs(np.fft.fftshift(uk) / len(uk))[N:]  # 1e6 for scaling to uv
        # Filter x and y for integer values of x from 1 to 20
        x_int = np.array([2 * k + 1 for k in range(21)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        ax.plot(x_int, y_int, color=color_list[i], marker='D', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(0, 20)
    ax.set_xticks(range(1, 21, 2))  # Set x-ticks every 2 units
    ax.set_ylabel('Harmonics Magnitude($\mu$v)', weight='bold', fontsize=20)
    ax.set_xlabel('f/fe', weight='bold', fontsize=20)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('SHS30-harmonics.png')

    # fwhm and peaks data for SHS30
    all_results = []
    for i in range(len(M)):
        Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He, dM[i], dH, params.nPeriod)
        resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
        resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
        combined_result = {'sigma': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('fwhm_SHS30.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    # SHP25 magnetization-time
    M = genfromtxt('SHP25.csv', delimiter=',')
    M_lowPass=np.zeros(M.shape)
    for i in range(M.shape[0]):
        M_lowPass[i,:] = low_pass_filter(M[i,:], 1, 10000*f, 3*f)
    stdSHP25 = .05
    sigmaCore_list = np.round(stdSHP25*np.array([.5, 1, 3, 5, 10]),3)
    color_list, trsp_list = colorMap(sigmaCore_list, 'red-brown', ['lightcoral', 'red', 'firebrick'] )
    fig, ax1 = initialize_figure()
    ax1.plot(t * 1e3, He * 1e3, 'black', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    for i in range(M.shape[0]):
        ax2.plot(t * 1e3, Ms*M_lowPass[i, :] * 1e-3 , linewidth=3.0, color=color_list[i], alpha=trsp_list[i], label=f'$\sigma$= {sigmaCore_list[i]}')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('SHP25-magnetization-time.png')

    # SHP25 Magnetization curve
    n, l = M.shape
    k = int(l / params.nPeriod)
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[-k:]* 1e3, Ms*M_lowPass[i, -k:] * 1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('SHP25-magnetization-curve.png')

    # SHP25 PSF
    dM = np.diff(np.append(M_lowPass, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, 0))
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[-k:]*1e3, dM[i, -k:]/dH[-k:] , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=20)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(-18,18)
    ax.set_ylim(0, 200)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('SHP25-psf.png')

    # SHP25 harmonics
    fig, ax = initialize_figure(figsize=(16,6))
    for i in range(M.shape[0]):
        dHz = (lz ** 3) * np.diff(np.append(u0 * Ms * M_lowPass[i, :], 0))
        uz = -pz * u0 * (dHz / dift)
        unet = uz - uz_free
        uk = np.fft.fft(unet)
        y = 1e6 * abs(np.fft.fftshift(uk) / len(uk))[N:]  # 1e6 for scaling to uv
        # Filter x and y for integer values of x from 1 to 20
        x_int = np.array([2 * k + 1 for k in range(21)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        ax.plot(x_int, y_int, color=color_list[i], marker='D', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(0, 20)
    ax.set_xticks(range(1, 21, 2))  # Set x-ticks every 2 units
    ax.set_ylabel('Harmonics Magnitude($\mu$v)', weight='bold', fontsize=20)
    ax.set_xlabel('f/fe', weight='bold', fontsize=20)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('SHP25-harmonics.png')

    # fwhm and peaks data for SHP25
    all_results = []
    for i in range(len(M)):
        Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He, dM[i], dH, params.nPeriod)
        resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
        resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
        combined_result = {'sigma': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('fwhm_SHP25.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    # SHP15 magnetization-time
    M = genfromtxt('SHP15.csv', delimiter=',')
    M_lowPass=np.zeros(M.shape)
    for i in range(M.shape[0]):
        M_lowPass[i,:] = low_pass_filter(M[i,:], 1, 10000*f, 3*f)
    stdSHP15 = .11
    sigmaCore_list = np.round(stdSHP15*np.array([.5, 1, 3, 5, 10]),3)
    color_list, trsp_list = colorMap(sigmaCore_list, 'grapes', ['violet', 'fuchsia', 'mediumvioletred'])
    fig, ax1 = initialize_figure()
    ax1.plot(t * 1e3, He * 1e3, 'black', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    for i in range(M.shape[0]):
        ax2.plot(t * 1e3, Ms*M_lowPass[i, :] * 1e-3 ,linewidth=3.0, color=color_list[i], alpha=trsp_list[i], label=f'$\sigma$= {sigmaCore_list[i]}')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('SHP15-magnetization-time.png')

    # SHP15 Magnetization curve
    n, l = M.shape
    k = int(l / params.nPeriod)
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[-k:]* 1e3, Ms*M_lowPass[i, -k:] * 1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('SHP15-magnetization-curve.png')

    # SHP15 PSF
    dM = np.diff(np.append(M_lowPass, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, 0))
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[-k:]*1e3, dM[i, -k:]/dH[-k:] , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=20)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(-18,18)
    ax.set_ylim(0, 50)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('SHP15-psf.png')

    # SHP15 harmonics
    fig, ax = initialize_figure(figsize=(16,6))
    for i in range(M.shape[0]):
        dHz = (lz ** 3) * np.diff(np.append(u0 * Ms * M_lowPass[i, :], 0))
        uz = -pz * u0 * (dHz / dift)
        unet = uz - uz_free
        uk = np.fft.fft(unet)
        y = 1e6 * abs(np.fft.fftshift(uk) / len(uk))[N:]  # 1e6 for scaling to uv
        # Filter x and y for integer values of x from 1 to 20
        x_int = np.array([2 * k + 1 for k in range(21)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        ax.plot(x_int, y_int, color=color_list[i], marker='D', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(0, 20)
    ax.set_xticks(range(1, 21, 2))  # Set x-ticks every 2 units
    ax.set_ylabel('Harmonics Magnitude($\mu$v)', weight='bold', fontsize=20)
    ax.set_xlabel('f/fe', weight='bold', fontsize=20)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('SHP15-harmonics.png')

    # fwhm and peaks data for SHP15
    all_results = []
    for i in range(len(M)):
        Hlmask, dMdHlmask, maskl, Hrmask, dMdHrmask, maskr = peaksInit(He, dM[i], dH, params.nPeriod)
        resultl = peaks_analysis(Hlmask, dMdHlmask, maskl)
        resultr = peaks_analysis(Hrmask, dMdHrmask, maskr)
        combined_result = {'sigma': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('fwhm_SHP15.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)


