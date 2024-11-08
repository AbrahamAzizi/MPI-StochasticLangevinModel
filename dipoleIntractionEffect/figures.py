from init import *
from utlis import Ht, low_pass_filter, peaks_analysis #,peaksInit
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.signal.windows import kaiser
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
    
    data = Data( )
    B = data.filedAmpl
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
    lz = B / gz
    dift = np.diff(t)
    dHz_free = (lz ** 3) * np.diff(Ht(f, t))
    uz_free = -pz * u0 * (dHz_free / dift)
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    N = len(freqs) // 2
    x = np.fft.fftshift(freqs / f)[N:]

    # this is equal with rsol since after offset length we have max field ( sin wave )
    # value at pi/4 and min field value at 2*pi/3 where the saturation occures
    winlen = (np.where(He == np.min(He))[0][0] - np.where(He == np.max(He))[0][0])
    print("window length: winlen=", winlen)
    wincnt = int(len(He)/winlen)
    print("number of windowed segments: wincnt=", wincnt)
    offlen = winlen//2 # to remove asyc part
    minDistList = np.array([50, 65, 100, 150, 200, 250])

    # Magnetization in time for core size 25 nm
    M = genfromtxt('dipoleIntractionEffect/data/M_size25.csv', delimiter=',')
    color_list, trsp_list = colorMap(minDistList, 'forest', ['lime', 'seagreen', 'forestgreen'])
    Mfilt=np.zeros(M.shape)
    for i in range(M.shape[0]):
        Mfilt[i,:] = low_pass_filter(Ms*M[i,:], 4, fs, 12*f)
    dco = 25e-9
    Vc = 1 / 6 * np.pi * dco ** 3
    mu = Ms * Vc

    fig, ax1 = initialize_figure()
    ax1.plot(t[offlen:]* 1e3, He[offlen:] * 1e3, '-k', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    for i in range(M.shape[0]):
        ax2.plot(t[offlen:]* 1e3, Mfilt[i, offlen:] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size25-time.png')

    # Magnetization curve for core size 25 nm
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        ax.plot(He[2*winlen:4*winlen] * 1e3, Mfilt[i, 2*winlen:4*winlen] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size25-magnetization.png')

    # psf for core size 25 nm
    psf = np.zeros(Mfilt.shape)
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(Mfilt.shape[0]):
        dM = np.diff(Mfilt[i, :])
        dH = np.diff(He)
        dMdH = dM/dH 
        for j in range(wincnt):
            tmp = dMdH[j*winlen : (j+1)*winlen]
            psf[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        ax.plot(He[2*winlen:4*winlen]*1e3, psf[i,2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size25-psf.png')

    # voltage core25
    uz = np.zeros(Mfilt.shape)
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(Mfilt.shape[0]):
        dM = (lz ** 3)*np.diff(u0 * Mfilt[i, :])
        dift = np.diff(t)
        dMdt = -pz * u0 * dM/dift 
        for j in range(wincnt):
            tmp = dMdt[j*winlen : (j+1)*winlen]
            uz[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        ax.plot(t*1e3, uz[i,:]*1e6, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax.set_ylabel('V(uv)', weight='bold', fontsize=30)
    ax.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size25-volt.png')

    # harmonics core size 25
    fig, ax = initialize_figure(figsize=(16,6))
    for i in range(M.shape[0]):
        uk = np.fft.fft(uz[i])
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        # Filter x and y for integer values of x from 1 to 20
        x_int = np.array([2*k+1 for k in range(1, 11)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        #markerline, stemlines, baseline = ax.stem(x_int, y_int, bottom=0, markerfmt="Dr")
        ax.plot(x_int, y_int, color=color_list[i], marker='D', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
        # Set color and alpha for stem lines
        #plt.setp(stemlines, color=color_list[i], alpha=trsp_list[i], linewidth=5)
        # Set color and alpha for markers
        #plt.setp(markerline, color=color_list[i], alpha=trsp_list[i], markersize=15)
        # Set color and alpha for baseline (usually invisible)
        #plt.setp(baseline, visible=False)
    ax.set_xlim(2, 12)
    ax.set_xticks(range(3, 13, 2))  # Set x-ticks every 2 units
    ax.set_ylabel(r'Harmonics Magnitude($\mu$v)', weight='bold', fontsize=20)
    ax.set_xlabel('Harmonics number', weight='bold', fontsize=20)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size25-harmonics.png')

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

    with open('dipoleIntractionEffect/data/fwhm_core25.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    # Magnetization in time for core size 30 nm
    M = genfromtxt('dipoleIntractionEffect/data/M_size30.csv', delimiter=',')
    minDistList = np.array([55, 70, 100, 150, 200, 250])
    color_list, trsp_list = colorMap(minDistList, 'red-brown', ['lightcoral', 'red', 'firebrick'] )
    M_filt=np.zeros(M.shape)
    for i in range(M.shape[0]):
        M_filt[i,:] = low_pass_filter(Ms*M[i,:], 4, fs, 12*f)
    dco = 30e-9
    Vc = 1 / 6 * np.pi * dco ** 3
    mu = Ms * Vc
    n, l = M.shape
    k = int(l / data.nPeriod)

    fig, ax1 = initialize_figure()
    ax1.plot(t[offlen:] * 1e3, He[offlen:] * 1e3, '-k', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    for i in range(M.shape[0]):
        ax2.plot(t[offlen:] * 1e3, M_filt[i, offlen:] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size30-time.png')

    # Magnetization curve for core size 30 nm
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        ax.plot(He[2*winlen:4*winlen] * 1e3, M_filt[i, 2*winlen:4*winlen] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size30-magnetization.png')

    # psf for core size 30 nm
    psf = np.zeros(Mfilt.shape)
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(Mfilt.shape[0]):
        dM = np.diff(Mfilt[i, :])
        dH = np.diff(He)
        dMdH = dM/dH 
        for j in range(wincnt-1):
            tmp = dMdH[j*winlen : (j+1)*winlen]
            psf[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        ax.plot(He[2*winlen:4*winlen]*1e3, psf[i,2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/CoreSize30-psf.png')

    # voltage core30
    uz = np.zeros(Mfilt.shape)
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(Mfilt.shape[0]):
        dM = (lz ** 3)*np.diff(u0 * Mfilt[i, :])
        dift = np.diff(t)
        dMdt = -pz * u0 * dM/dift 
        for j in range(wincnt):
            tmp = dMdt[j*winlen : (j+1)*winlen]
            uz[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        ax.plot(t*1e3, uz[i,:]*1e6, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax.set_ylabel('V(uv)', weight='bold', fontsize=30)
    ax.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size30-volt.png')

    # harmonics core size 30
    fig, ax = initialize_figure(figsize=(16,6))
    for i in range(M.shape[0]):
        uk = np.fft.fft(uz[i])
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        x_int = np.array([2*k+1 for k in range(1, 11)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        ax.plot(x_int, y_int, color=color_list[i], marker='D', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(2, 12)
    ax.set_xticks(range(3, 13, 2))  # Set x-ticks every 2 units
    ax.set_ylabel(r'Harmonics Magnitude($\mu$v)', weight='bold', fontsize=20)
    ax.set_xlabel('Harmonics number', weight='bold', fontsize=20)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/CoreSize30-harmonics.png')

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

    with open('dipoleIntractionEffect/data/fwhm_core30.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    # Magnetization in time for core size 35 nm
    M = genfromtxt('dipoleIntractionEffect/data/M_size35.csv', delimiter=',')
    minDistList = np.array([60, 75, 100, 150, 200, 250])
    color_list, trsp_list = colorMap(minDistList, 'sunset', ['gold', 'orange', 'darkorange'])
    M_filt=np.zeros(M.shape)
    for i in range(M.shape[0]):
        M_filt[i,:] = low_pass_filter(Ms*M[i,:], 4, fs, 12*f)
    dco = 35e-9
    Vc = 1 / 6 * np.pi * dco ** 3
    mu = Ms * Vc
    n, l = M.shape
    k = int(l / data.nPeriod)

    fig, ax1 = initialize_figure()
    ax1.plot(t[offlen:] * 1e3, He[offlen:] * 1e3, '-k', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    for i in range(M.shape[0]):
        ax2.plot(t[offlen:] * 1e3, M_filt[i, offlen:] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size35-time.png')

    # Magnetization curve for core size 35 nm
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        ax.plot(He[2*winlen:4*winlen] * 1e3, M_filt[i, 2*winlen:4*winlen] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size35-magnetization.png')

    # psf for core size 35 nm
    psf = np.zeros(Mfilt.shape)
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(Mfilt.shape[0]):
        dM = np.diff(Mfilt[i, :])
        dH = np.diff(He)
        dMdH = dM/dH 
        for j in range(wincnt):
            tmp = dMdH[j*winlen : (j+1)*winlen]
            psf[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        ax.plot(He[2*winlen:4*winlen]*1e3, psf[i,2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size35-psf.png')

    # voltage core35
    uz = np.zeros(Mfilt.shape)
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(Mfilt.shape[0]):
        dM = (lz ** 3)*np.diff(u0 * Mfilt[i, :])
        dift = np.diff(t)
        dMdt = -pz * u0 * dM/dift 
        for j in range(wincnt):
            tmp = dMdt[j*winlen : (j+1)*winlen]
            uz[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 40, 2, mode='nearest')
        ax.plot(t*1e3, uz[i,:]*1e6, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax.set_ylabel('V(uv)', weight='bold', fontsize=30)
    ax.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size35-volt.png')

    # harmonics core35
    fig, ax = initialize_figure(figsize=(16,6))
    for i in range(M.shape[0]):
        uk = np.fft.fft(uz[i])
        y = 1e6 * abs(np.fft.fftshift(uk) / len(uk))[N:]  # 1e6 for scaling to uv
        # Filter x and y for integer values of x from 1 to 20
        x_int = np.array([2 * k + 1 for k in range(1, 11)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        ax.plot(x_int, y_int, color=color_list[i], marker='D', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(2, 12)
    ax.set_xticks(range(3, 13, 2))  # Set x-ticks every 2 units
    ax.set_ylabel(r'Harmonics Magnitude($\mu$v)', weight='bold', fontsize=20)
    ax.set_xlabel('Harmonics number', weight='bold', fontsize=20)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size35-harmonics.png')

   # fwhm and peaks data for core 35
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

    with open('dipoleIntractionEffect/data/fwhm_core35.csv', 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)





