from init import *
from utlis import low_pass_filter, peaks_analysis, Ht
import numpy as np
from numpy import genfromtxt
from scipy.signal import savgol_filter
from scipy.signal.windows import kaiser
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
import seaborn as sns
import csv

def initialize_figure(figsize=(22,6), dpi=300, font_scale=2):
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
    
    data = Data( )
    B = data.fieldAmpl
    f = data.fieldFreq
    Ms = data.Ms
    cycs = data.nPeriod
    data.rsol = 100
    rsol = data.rsol
    num = data.nParticle
    beta = 14 # for kasier

    fs = rsol*2*f
    dt = 1/fs
    tf = cycs*(1/f)
    lent = int(np.ceil(tf/dt))

    t = np.array([i*dt for i in range(lent)])
    He = np.array([B*Ht(f,i*dt) for i in range(lent)])

    # for induced voltages and harmonics
    u0 = 4 * np.pi * 1e-7  # T.m/A, v.s/A/m
    gz = 3  # T/m
    pz = 20e-3 * 795.7747 / 1.59  # A/m/A   1T = 795.7747 A/m, I = 1.59 A
    #lz = B / gz ffp volume
    dift = np.diff(t)
    freqs = np.fft.fftfreq(lent, dt)
    N = len(freqs) // 2
    x = np.fft.fftshift(freqs / f)[N:]

    winlen = (np.where(He == np.min(He))[0][0] - np.where(He == np.max(He))[0][0])
    print("window length: winlen=", winlen)
    wincnt = int(len(He)/winlen)
    print("number of windowed segments: wincnt=", wincnt)
    offlen = winlen//2 # to remove asyc part

    sumv = genfromtxt('MNP-sizeDistribution/data/sumVc.csv', delimiter=',')

    # IGP30 magnetization-time
    M = genfromtxt('MNP-sizeDistribution/data/IPG30.csv', delimiter=',')
    meanCoreIGP30 = 29.53*1e-9
    meanVolIGP30 = 1 / 6 * np.pi * meanCoreIGP30 ** 3
    stdIGP30 = .09
    sigmaCore_list = np.round(stdIGP30*np.array([.5, 1, 3, 5, 10]),3)
    color_list, trsp_list = colorMap(sigmaCore_list, 'forest', ['lime', 'seagreen', 'forestgreen'])
    fig, ax1 = initialize_figure()
    ax1.plot(t * 1e3, He * 1e3, 'black', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    ax1.set_xlim(.01, .11)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    for i in range(M.shape[0]):
        ax2.plot(t * 1e3, Ms*M[i, :] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=rf'$\sigma$= {sigmaCore_list[i]}')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax2.xaxis.set_tick_params(labelsize=30)
    ax2.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/IGP30-magnetization-time.png')

    # IPG30 Magnetization curve
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[4*winlen:]* 1e3, Ms*M[i, 4*winlen:] * 1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/IPG30-magnetization-curve.png')

    # IPG30 harmonics
    fig, ax = initialize_figure(figsize=(16,6))
    uz = np.zeros(M.shape)
    for i in range(M.shape[0]):
        dM = sumv[0]*np.diff(u0 * Ms*M[i, :])
        dift = np.diff(t)
        dMdt = -pz * u0 * dM/dift 
        for j in range(wincnt):
            tmp = dMdt[j*winlen : (j+1)*winlen]
            uz[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        uk = np.fft.fft(uz[i])
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        x_int = np.array([2*k+1 for k in range(1, 11)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        ax.plot(x_int, y_int, color=color_list[i], marker='8', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(2, 12)
    ax.set_xticks(range(3, 13, 2))  # Set x-ticks every 2 units
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.set_ylabel(r'Magnitude($\mu$v)', weight='bold', fontsize=30)
    ax.set_xlabel('Harmonics number', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/IPG30-harmonics.png')

    # IPG30 PSF
    fig, ax = initialize_figure(figsize=(12,6))
    psf = np.zeros(M.shape)
    for i in range(M.shape[0]):
        dM = np.diff(Ms*M[i, :])
        dH = np.diff(He)
        dMdH = dM/dH 
        for j in range(wincnt):
            tmp = dMdH[j*winlen : (j+1)*winlen]
            psf[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        ax.plot(He[2*winlen:4*winlen]*1e3, psf[i,2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/IPG30-psf.png')

    # fwhm and peaks data for IPG30
    all_results = []
    for i in range(M.shape[0]):
        maskl = np.where((He[2*winlen:3*winlen] >= -B) & (He[2*winlen:3*winlen]<= B))[0]
        resultl = peaks_analysis(He[2*winlen:3*winlen], psf[i, 2*winlen:3*winlen], maskl)
        maskr = np.where((He[3*winlen:4*winlen] >= -B) & (He[3*winlen:4*winlen]<= B))[0]
        resultr = peaks_analysis(He[3*winlen:4*winlen], psf[i, 3*winlen:4*winlen], maskr)
        combined_result = {'sigma': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('MNP-sizeDistribution/data/fwhmIPG30.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    # SHS30 magnetization-time
    M = genfromtxt('MNP-sizeDistribution/data/SHS30.csv', delimiter=',')
    meanCoreSHS30 = 29.53*1e-9
    meanVolSHS30= 1 / 6 * np.pi * meanCoreSHS30 ** 3
    stdSHS30 = .08
    sigmaCore_list = np.round(stdSHS30*np.array([.5, 1, 3, 5, 10]),3)
    color_list, trsp_list = colorMap(sigmaCore_list, 'sunset', ['gold', 'orange', 'darkorange'])
    fig, ax1 = initialize_figure()
    ax1.plot(t * 1e3, He * 1e3, 'black', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=30)
    ax2.yaxis.set_tick_params(labelsize=30)
    for i in range(M.shape[0]):
        ax2.plot(t * 1e3, Ms*M[i, :] * 1e-3 ,linewidth=3.0, color=color_list[i], alpha=trsp_list[i], label=rf'$\sigma$= {sigmaCore_list[i]}')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.set_xlim(.01, .11)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/SHS30-magnetization-time.png')

    # SHS30 Magnetization curve
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[4*winlen:]* 1e3, Ms*M[i, 4*winlen:] * 1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel(r'Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/SHS30-magnetization-curve.png')

    # SHS30 harmonics
    fig, ax = initialize_figure(figsize=(16,6))
    uz = np.zeros(M.shape)
    for i in range(M.shape[0]):
        dM = sumv[1]*np.diff(u0 * Ms*M[i, :])
        dift = np.diff(t)
        dMdt = -pz * u0 * dM/dift 
        for j in range(wincnt):
            tmp = dMdt[j*winlen : (j+1)*winlen]
            uz[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        uk = np.fft.fft(uz[i])
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        x_int = np.array([2*k+1 for k in range(1, 11)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        ax.plot(x_int, y_int, color=color_list[i], marker='8', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(2, 12)
    ax.set_xticks(range(3, 13, 2))  # Set x-ticks every 2 units
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.set_ylabel(r'Magnitude($\mu$v)', weight='bold', fontsize=30)
    ax.set_xlabel('Harmonics number', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/SHS30-harmonics.png')

    # SHS30 PSF
    fig, ax = initialize_figure(figsize=(12,6))
    psf = np.zeros(M.shape)
    for i in range(M.shape[0]):
        dM = np.diff(Ms*M[i, :])
        dH = np.diff(He)
        dMdH = dM/dH 
        for j in range(wincnt):
            tmp = dMdH[j*winlen : (j+1)*winlen]
            psf[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        ax.plot(He[2*winlen:4*winlen]*1e3, psf[i,2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/SHS30-psf.png')

    # fwhm and peaks data for core 30
    all_results = []
    for i in range(len(M)):
        maskl = np.where((He[2*winlen:3*winlen] >= -B) & (He[2*winlen:3*winlen]<= B))[0]
        resultl = peaks_analysis(He[2*winlen:3*winlen], psf[i, 2*winlen:3*winlen], maskl)
        maskr = np.where((He[3*winlen:4*winlen] >= -B) & (He[3*winlen:4*winlen]<= B))[0]
        resultr = peaks_analysis(He[3*winlen:4*winlen], psf[i, 3*winlen:4*winlen], maskr)
        combined_result = {'sigma': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('MNP-sizeDistribution/data/fwhmSHP30.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    # SHP25 magnetization-time
    M = genfromtxt('MNP-sizeDistribution/data/SHP25.csv', delimiter=',')
    meanCoreSHP25 = 29.53*1e-9
    meanVolHP25 = 1 / 6 * np.pi * meanCoreSHP25 ** 3
    stdSHP25 = .05
    sigmaCore_list = np.round(stdSHP25*np.array([.5, 1, 3, 5, 10]),3)
    color_list, trsp_list = colorMap(sigmaCore_list, 'red-brown', ['lightcoral', 'red', 'firebrick'] )
    fig, ax1 = initialize_figure()
    ax1.plot(t * 1e3, He * 1e3, 'black', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=30)
    ax2.yaxis.set_tick_params(labelsize=30)
    for i in range(M.shape[0]):
        ax2.plot(t * 1e3, Ms*M[i, :] * 1e-3 , linewidth=3.0, color=color_list[i], alpha=trsp_list[i], label=rf'$\sigma$= {sigmaCore_list[i]}')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax2.set_xlim(.01, .11)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/SHP25-magnetization-time.png')

    # SHP25 Magnetization curve
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[4*winlen:]* 1e3, Ms*M[i, 4*winlen:] * 1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/SHP25-magnetization-curve.png')

    # SHP25 harmonics
    fig, ax = initialize_figure(figsize=(16,6))
    uz = np.zeros(M.shape)
    for i in range(M.shape[0]):
        dM = sumv[2]*np.diff(u0 * Ms*M[i, :])
        dift = np.diff(t)
        dMdt = -pz * u0 * dM/dift 
        for j in range(wincnt):
            tmp = dMdt[j*winlen : (j+1)*winlen]
            uz[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        uk = np.fft.fft(uz[i])
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        x_int = np.array([2*k+1 for k in range(1, 11)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        ax.plot(x_int, y_int, color=color_list[i], marker='8', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(2, 12)
    ax.set_xticks(range(3, 13, 2))  # Set x-ticks every 2 units
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.set_ylabel(r'Magnitude($\mu$v)', weight='bold', fontsize=30)
    ax.set_xlabel('Harmonics number', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/SHP25-harmonics.png')

    # SHP25 PSF
    fig, ax = initialize_figure(figsize=(12,6))
    psf = np.zeros(M.shape)
    for i in range(M.shape[0]):
        dM = np.diff(Ms*M[i, :])
        dH = np.diff(He)
        dMdH = dM/dH 
        for j in range(wincnt):
            tmp = dMdH[j*winlen : (j+1)*winlen]
            psf[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        ax.plot(He[2*winlen:4*winlen]*1e3, psf[i,2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/SHP25-psf.png')

    # fwhm and peaks data for core 25
    all_results = []
    for i in range(len(M)):
        maskl = np.where((He[2*winlen:3*winlen] >= -B) & (He[2*winlen:3*winlen]<= B))[0]
        resultl = peaks_analysis(He[2*winlen:3*winlen], psf[i, 2*winlen:3*winlen], maskl)
        maskr = np.where((He[3*winlen:4*winlen] >= -B) & (He[3*winlen:4*winlen]<= B))[0]
        resultr = peaks_analysis(He[3*winlen:4*winlen], psf[i, 3*winlen:4*winlen], maskr)
        combined_result = {'sigma': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('MNP-sizeDistribution/data/fwhmSHP25.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    # SHP15 magnetization-time
    M = genfromtxt('MNP-sizeDistribution/data/SHP15.csv', delimiter=',')
    meanCoreSHP15 = 29.53*1e-9
    meanVolSHP15 = 1 / 6 * np.pi * meanCoreSHP15 ** 3
    stdSHP15 = .11
    sigmaCore_list = np.round(stdSHP15*np.array([.5, 1, 3, 5, 10]),3)
    color_list, trsp_list = colorMap(sigmaCore_list, 'grapes', ['violet', 'fuchsia', 'mediumvioletred'])
    fig, ax1 = initialize_figure()
    ax1.plot(t * 1e3, He * 1e3, 'black', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    ax1.set_xlim(.01, .11)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=30)
    ax2.yaxis.set_tick_params(labelsize=30)
    for i in range(M.shape[0]):
        ax2.plot(t * 1e3, Ms*M[i, :] * 1e-3 ,linewidth=3.0, color=color_list[i], alpha=trsp_list[i], label=rf'$\sigma$= {sigmaCore_list[i]}')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/SHP15-magnetization-time.png')

    # SHP15 Magnetization curve
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[4*winlen:]* 1e3, Ms*M[i, 4*winlen:] * 1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/SHP15-magnetization-curve.png')

    # SHP15 harmonics
    fig, ax = initialize_figure(figsize=(16,6))
    uz = np.zeros(M.shape)
    for i in range(M.shape[0]):
        dM = sumv[3]*np.diff(u0 * Ms*M[i, :])
        dift = np.diff(t)
        dMdt = -pz * u0 * dM/dift 
        for j in range(wincnt):
            tmp = dMdt[j*winlen : (j+1)*winlen]
            uz[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        uk = np.fft.fft(uz[i])
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        x_int = np.array([2*k+1 for k in range(1, 11)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        ax.plot(x_int, y_int, color=color_list[i], marker='8', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(2, 12)
    ax.set_xticks(range(3, 13, 2))  # Set x-ticks every 2 units
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.set_ylabel(r'Magnitude($\mu$v)', weight='bold', fontsize=30)
    ax.set_xlabel('Harmonics number', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/SHP15-harmonics.png')

    # SHP15 PSF
    fig, ax = initialize_figure(figsize=(12,6))
    psf = np.zeros(M.shape)
    for i in range(M.shape[0]):
        dM = np.diff(Ms*M[i, :])
        dH = np.diff(He)
        dMdH = dM/dH 
        for j in range(wincnt):
            tmp = dMdH[j*winlen : (j+1)*winlen]
            psf[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        ax.plot(He[2*winlen:4*winlen]*1e3, psf[i,2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('MNP-sizeDistribution/figures/SHP15-psf.png')

    # fwhm and peaks data for core 15
    all_results = []
    for i in range(len(M)):
        maskl = np.where((He[2*winlen:3*winlen] >= -B) & (He[2*winlen:3*winlen]<= B))[0]
        resultl = peaks_analysis(He[2*winlen:3*winlen], psf[i, 2*winlen:3*winlen], maskl)
        maskr = np.where((He[3*winlen:4*winlen] >= -B) & (He[3*winlen:4*winlen]<= B))[0]
        resultr = peaks_analysis(He[3*winlen:4*winlen], psf[i, 3*winlen:4*winlen], maskr)
        combined_result = {'sigma': i}  # Include the index as a core list identifier
        combined_result.update({f'left peak {key}': value for key, value in resultl.items()})
        combined_result.update({f'right peak {key}': value for key, value in resultr.items()})
        all_results.append(combined_result)

    with open('MNP-sizeDistribution/data/fwhmSHP15.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)


