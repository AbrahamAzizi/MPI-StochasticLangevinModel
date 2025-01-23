from init import *
from utlis import Ht, low_pass_filter, peaks_analysis #,peaksInit
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.signal.windows import kaiser
import csv

### SetFont
path = 'arial/'
font_files = []
# load fonts
font_files.extend(fm.findSystemFonts(fontpaths=path))
for font_file in font_files:  # add fonts to fontmanager
    fm.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Arial'   # set font
###

def initialize_figure(figsize=(18, 12), dpi=300, font_scale=2):
    sns.set_context("notebook", font_scale=font_scale)
    sns.set_style("whitegrid")
    rcParams['font.weight'] = 'bold'
    fig = plt.figure(figsize=figsize, dpi=dpi)
    return fig

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
    kT = data.kB*data.temp
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
    pz = 1e-3 # T/A
    dift = np.diff(t)
    freqs = np.fft.fftfreq(lent, dt)
    N = len(freqs) // 2
    x = np.fft.fftshift(freqs / f)[N:]

    # this is equal with rsol since after offset length we have max field ( sin wave )
    # value at pi/4 and min field value at 2*pi/3 where the saturation occures
    winlen = (np.where(He == np.min(He))[0][0] - np.where(He == np.max(He))[0][0])
    print("window length: winlen=", winlen)
    wincnt = int(len(He)/winlen)
    print("number of windowed segments: wincnt=", wincnt)
    offlen = winlen//2 # to remove asyc part

    # size 25 nm
    minDistList = np.array([50, 65, 100, 125, 150, 175, 200, 250])
    M = genfromtxt('dipoleIntractionEffect/data/M_size25.csv', delimiter=',')
    color_list, trsp_list = colorMap(minDistList, 'forest', ['lime', 'seagreen', 'forestgreen'])
    Mfilt=np.zeros(M.shape)
    for i in range(M.shape[0]):
        Mfilt[i,:] = low_pass_filter(M[i,:], 4, fs, 15*f)
    dco = 25e-9
    Vc = 1 / 6 * np.pi * dco ** 3
    mu = Ms * Vc

    fig = initialize_figure()
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1])

    # size25: M-t
    ax = fig.add_subplot(gs[0, 0])
    for i in range(M.shape[0]):
        ax.plot(t* 1e3, Ms*Mfilt[i] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$d = $ {minDistList[i]} nm')
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax)
    ax1 = ax.twinx()
    ax1.plot(t* 1e3, He * 1e3, '-k', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax1)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax1)

    # size25: min distance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off') 
    lines, labels = ax1.get_legend_handles_labels() # H (ax1) as the first in legend
    lines2, labels2 = ax.get_legend_handles_labels()
    legend = ax2.legend(lines + lines2, labels + labels2, loc='center', prop={'size': 30})
    set_legend_properties(legend)
    set_spines_grid(ax2)

    # size25: voltage
    ax3 = fig.add_subplot(gs[1, 0])
    uz = np.zeros(Mfilt.shape)
    for i in range(Mfilt.shape[0]):
        dM = np.diff(Mfilt[i, :])
        dift = np.diff(t)
        dMdt = -(pz*mu*num)* dM/dift 
        for j in range(wincnt):
            tmp = dMdt[j*winlen : (j+1)*winlen]
            uz[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 20, 1, mode='nearest')
        ax3.plot(t*1e3, uz[i,:]*1e6, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$d = $ {minDistList[i]} nm')
    ax3.xaxis.set_tick_params(labelsize=30)
    ax3.yaxis.set_tick_params(labelsize=30)
    ax3.set_ylabel('V(uv)', weight='bold', fontsize=30)
    ax3.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    set_spines_grid(ax3)

    # size25: harmonics
    ax4 = fig.add_subplot(gs[1, 1])
    for i in range(M.shape[0]):
        uk = np.fft.fft(uz[i])
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        x_int = np.array([2*k+1 for k in range(1, 11)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        ax4.plot(x_int, y_int, color=color_list[i], marker='8', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax4.set_xlim(2, 12)
    ax4.set_xticks(range(3, 13, 2))  # Set x-ticks every 2 units
    ax4.xaxis.set_tick_params(labelsize=30)
    ax4.yaxis.set_tick_params(labelsize=30)
    ax4.set_ylabel(r'Harmonics Magnitude($\mu$v)', weight='bold', fontsize=30)
    ax4.set_xlabel('Harmonics number', weight='bold', fontsize=30)
    set_spines_grid(ax4)

    plt.tight_layout()

    plt.savefig('dipoleIntractionEffect/figures/size25-tf.png')

    # size25: magnetization curve
    fig = initialize_figure(figsize=(12,6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0])
    for i in range(M.shape[0]):
        ax.plot(He[2*winlen:4*winlen] * 1e3, Mfilt[i, 2*winlen:4*winlen] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax)

    # psf for core size 25 nm
    ax1 = fig.add_subplot(gs[0, 1])
    psf = np.zeros(Mfilt.shape)
    for i in range(Mfilt.shape[0]):
        dM = np.diff(Mfilt[i, :])
        dH = np.diff(He)
        dMdH = Ms*dM/dH 
        for j in range(wincnt):
            tmp = dMdH[j*winlen : (j+1)*winlen]
            psf[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 20, 1, mode='nearest')
        ax1.plot(He[2*winlen:4*winlen]*1e3, psf[i,2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$d = $ {minDistList[i]} nm')
    ax1.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax1.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax1)

    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size25-psf.png')

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

    # size 30 nm
    minDistList = np.array([55, 70, 100, 125, 150, 175, 200, 250])
    M = genfromtxt('dipoleIntractionEffect/data/M_size30.csv', delimiter=',')
    color_list, trsp_list = colorMap(minDistList, 'red-brown', ['lightcoral', 'red', 'firebrick'] )
    Mfilt=np.zeros(M.shape)
    for i in range(M.shape[0]):
        Mfilt[i,:] = low_pass_filter(Ms*M[i,:], 4, fs, 15*f)
    dco = 30e-9
    Vc = 1 / 6 * np.pi * dco ** 3
    mu = Ms * Vc
    n, l = M.shape
    k = int(l / data.nPeriod)

    fig = initialize_figure()
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1])

    # size30: M-t
    ax = fig.add_subplot(gs[0, 0])
    for i in range(M.shape[0]):
        ax.plot(t* 1e3, Mfilt[i] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$d = $ {minDistList[i]} nm')
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax)
    ax1 = ax.twinx()
    ax1.plot(t* 1e3, He * 1e3, '-k', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax1)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax1)

    # size30: min distance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off') 
    lines, labels = ax1.get_legend_handles_labels() # H (ax1) as the first in legend
    lines2, labels2 = ax.get_legend_handles_labels()
    legend = ax2.legend(lines + lines2, labels + labels2, loc='center', prop={'size': 30})
    set_legend_properties(legend)
    set_spines_grid(ax2)

    # size30: voltage
    ax3 = fig.add_subplot(gs[1, 0])
    uz = np.zeros(Mfilt.shape)
    for i in range(Mfilt.shape[0]):
        dM = np.diff(u0 * Mfilt[i, :])
        dift = np.diff(t)
        dMdt = -(pz*mu*num)* dM/dift 
        for j in range(wincnt):
            tmp = dMdt[j*winlen : (j+1)*winlen]
            uz[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 20, 1, mode='nearest')
        ax3.plot(t*1e3, uz[i,:]*1e6, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$d = $ {minDistList[i]} nm')
    ax3.xaxis.set_tick_params(labelsize=30)
    ax3.yaxis.set_tick_params(labelsize=30)
    ax3.set_ylabel('V(uv)', weight='bold', fontsize=30)
    ax3.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    set_spines_grid(ax3)

    # size30: harmonics
    ax4 = fig.add_subplot(gs[1, 1])
    for i in range(M.shape[0]):
        uk = np.fft.fft(uz[i])
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        # Filter x and y for integer values of x from 1 to 20
        x_int = np.array([2*k+1 for k in range(1, 11)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        #markerline, stemlines, baseline = ax.stem(x_int, y_int, bottom=0, markerfmt="Dr")
        ax4.plot(x_int, y_int, color=color_list[i], marker='8', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
        # Set color and alpha for stem lines
        #plt.setp(stemlines, color=color_list[i], alpha=trsp_list[i], linewidth=5)
        # Set color and alpha for markers
        #plt.setp(markerline, color=color_list[i], alpha=trsp_list[i], markersize=15)
        # Set color and alpha for baseline (usually invisible)
        #plt.setp(baseline, visible=False)
    ax4.set_xlim(2, 12)
    ax4.set_xticks(range(3, 13, 2))  # Set x-ticks every 2 units
    ax4.xaxis.set_tick_params(labelsize=30)
    ax4.yaxis.set_tick_params(labelsize=30)
    ax4.set_ylabel(r'Harmonics Magnitude($\mu$v)', weight='bold', fontsize=30)
    ax4.set_xlabel('Harmonics number', weight='bold', fontsize=30)
    set_spines_grid(ax4)

    plt.tight_layout()

    plt.savefig('dipoleIntractionEffect/figures/size30-tf.png')

    fig = initialize_figure(figsize=(12,6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

    #size30: magnetization curve
    ax = fig.add_subplot(gs[0, 0])
    for i in range(M.shape[0]):
        ax.plot(He[2*winlen:4*winlen] * 1e3, Mfilt[i, 2*winlen:4*winlen] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax)

    # psf for core size 30 nm
    ax1 = fig.add_subplot(gs[0, 1])
    psf = np.zeros(Mfilt.shape)
    for i in range(Mfilt.shape[0]):
        dM = np.diff(Mfilt[i, :])
        dH = np.diff(He)
        dMdH = Ms*dM/dH 
        for j in range(wincnt):
            tmp = dMdH[j*winlen : (j+1)*winlen]
            psf[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 20, 1, mode='nearest')
        ax1.plot(He[2*winlen:4*winlen]*1e3, psf[i,2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$d = $ {minDistList[i]} nm')
    ax1.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax1.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax1)

    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size30-psf.png')

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

    # size 35 nm
    M = genfromtxt('dipoleIntractionEffect/data/M_size35.csv', delimiter=',')
    minDistList = np.array([60, 75, 100, 125, 150, 175, 200, 250])
    color_list, trsp_list = colorMap(minDistList, 'sunset', ['gold', 'orange', 'darkorange'])
    Mfilt=np.zeros(M.shape)
    for i in range(M.shape[0]):
        Mfilt[i,:] = low_pass_filter(Ms*M[i,:], 4, fs, 15*f)
    dco = 35e-9
    Vc = 1 / 6 * np.pi * dco ** 3
    mu = Ms * Vc
    n, l = M.shape
    k = int(l / data.nPeriod)

    fig = initialize_figure()
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1])

    # size35: M-t
    ax = fig.add_subplot(gs[0, 0])
    for i in range(M.shape[0]):
        ax.plot(t* 1e3, Mfilt[i] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$d = $ {minDistList[i]} nm')
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax)
    ax1 = ax.twinx()
    ax1.plot(t* 1e3, He * 1e3, '-k', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax1)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax1)

    # size35: min distance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off') 
    lines, labels = ax1.get_legend_handles_labels() # H (ax1) as the first in legend
    lines2, labels2 = ax.get_legend_handles_labels()
    legend = ax2.legend(lines + lines2, labels + labels2, loc='center', prop={'size': 30})
    set_legend_properties(legend)
    set_spines_grid(ax2)

    # size35: voltage
    ax3 = fig.add_subplot(gs[1, 0])
    uz = np.zeros(Mfilt.shape)
    for i in range(Mfilt.shape[0]):
        dM = np.diff(u0 * Mfilt[i, :])
        dift = np.diff(t)
        dMdt = -(pz*mu*num)* dM/dift 
        for j in range(wincnt):
            tmp = dMdt[j*winlen : (j+1)*winlen]
            uz[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 20, 1, mode='nearest')
        ax3.plot(t*1e3, uz[i,:]*1e6, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$d = $ {minDistList[i]} nm')
    ax3.xaxis.set_tick_params(labelsize=30)
    ax3.yaxis.set_tick_params(labelsize=30)
    ax3.set_ylabel('V(uv)', weight='bold', fontsize=30)
    ax3.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    set_spines_grid(ax3)

    # size35: harmonics
    ax4 = fig.add_subplot(gs[1, 1])
    for i in range(M.shape[0]):
        uk = np.fft.fft(uz[i])
        y = 1e6*abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        # Filter x and y for integer values of x from 1 to 20
        x_int = np.array([2*k+1 for k in range(1, 11)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        #markerline, stemlines, baseline = ax.stem(x_int, y_int, bottom=0, markerfmt="Dr")
        ax4.plot(x_int, y_int, color=color_list[i], marker='8', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
        # Set color and alpha for stem lines
        #plt.setp(stemlines, color=color_list[i], alpha=trsp_list[i], linewidth=5)
        # Set color and alpha for markers
        #plt.setp(markerline, color=color_list[i], alpha=trsp_list[i], markersize=15)
        # Set color and alpha for baseline (usually invisible)
        #plt.setp(baseline, visible=False)
    ax4.set_xlim(2, 12)
    ax4.set_xticks(range(3, 13, 2))  # Set x-ticks every 2 units
    ax4.xaxis.set_tick_params(labelsize=30)
    ax4.yaxis.set_tick_params(labelsize=30)
    ax4.set_ylabel(r'Harmonics Magnitude($\mu$v)', weight='bold', fontsize=30)
    ax4.set_xlabel('Harmonics number', weight='bold', fontsize=30)
    set_spines_grid(ax4)

    plt.tight_layout()

    plt.savefig('dipoleIntractionEffect/figures/size35-tf.png')

    fig = initialize_figure(figsize=(12,6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

    # magnetization curve size35
    ax = fig.add_subplot(gs[0, 0])
    for i in range(M.shape[0]):
        ax.plot(He[2*winlen:4*winlen] * 1e3, Mfilt[i, 2*winlen:4*winlen] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$min distance$ {minDistList[i]} nm')
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax)

    # psf for core size 35 nm
    ax1 = fig.add_subplot(gs[0, 1])
    psf = np.zeros(Mfilt.shape)
    for i in range(Mfilt.shape[0]):
        dM = np.diff(Mfilt[i, :])
        dH = np.diff(He)
        dMdH = Ms*dM/dH 
        for j in range(wincnt):
            tmp = dMdH[j*winlen : (j+1)*winlen]
            psf[ i ,j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,beta)*tmp, 20, 1, mode='nearest')
        ax1.plot(He[2*winlen:4*winlen]*1e3, psf[i,2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$d = $ {minDistList[i]} nm')
    ax1.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax1.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax1)

    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size35-psf.png')

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


    # size 35 nm
    Hd = genfromtxt('dipoleIntractionEffect/data/Hdd_size35.csv', delimiter=',')
    color_list1, trsp_list = colorMap(minDistList, 'forest', ['lime', 'seagreen', 'forestgreen'])
    color_list2, trsp_list = colorMap(minDistList, 'sunset', ['gold', 'orange', 'darkorange'])

    fig = initialize_figure(figsize=(18,6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

    # size35: M & Hdd - t 
    ax = fig.add_subplot(gs[0, 0])
    ax1 = ax.twinx()
    for i in range(Hd.shape[0]):
        ax.plot(t* 1e3, Hd[i] * 1e-3, color=color_list1[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$d=$ {minDistList[i]} nm')
        ax1.plot(t* 1e3, M[i] * 1e-3, color=color_list2[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Hdd (kA/m)', weight='bold', fontsize=30)
    ax1.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax)
    set_spines_grid(ax1)

    # size35: min distance for legends
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off') 
    lines, labels = ax.get_legend_handles_labels() # H (ax1) as the first in legend
    legend = ax2.legend(lines, labels, loc='center', prop={'size': 30})
    set_legend_properties(legend)
    set_spines_grid(ax2)   

    plt.tight_layout()
    plt.savefig('dipoleIntractionEffect/figures/size35-Hd.png')

