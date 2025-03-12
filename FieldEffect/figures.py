from init import *
from utils import *
import numpy as np
from numpy import genfromtxt
from scipy.signal import savgol_filter
from scipy.signal.windows import kaiser
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

def plot_M_t(t, He, m, Ms, color_list, trsp_list, figName):
    _, ax1 = initialize_figure()
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.set_xlim(.01, .11)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    for i in range(m.shape[0]):
        ax1.plot(t * 1e3, He[i] * 1e3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$\mu_0H$= {fieldAml_list[i]} mT')
        ax2.plot(t * 1e3, Ms*m[i, :] * 1e-3 , '--', color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=fr'$M_z$ at {fieldAml_list[i]} mT')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.set_xlim(.01, .11)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig(f'FieldEffect/figures/{figName}_M_t.png')

def plot_M_H(He, m, Ms, winlen, color_list, trsp_list, figName):
    _, ax = initialize_figure(figsize=(12,6))
    for i in range(m.shape[0]):
        ax.plot(He[i, 2*winlen:4*winlen]*1e3, Ms*m[i, 2*winlen:4*winlen]*1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'FieldEffect/figures/{figName}_M_H.png')

def plot_Signal(xiH, sigH, m, lz, dt, f, num, mu, lent, pz, cycs, figName):
    _, ax = initialize_figure(figsize=(12,6))
    sf = np.zeros((len(fieldAml_list), lent))
    for i in range(len(fieldAml_list)):
        st, sf[i] = ftsignal(xiH[i], sigH[i], m[i], lz, dt, num, mu, lent, pz, cycs)
        ax.plot(t*1e3, st, color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('V/m', weight='bold', fontsize=30)
    ax.set_xlabel('t(ms)', weight='bold', fontsize=30)
    ax.set_xlim(.01, .11)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'FieldEffect/figures/{figName}_Signals.png')
    #plot_Harmonics
    freqs = np.fft.fftfreq(lent, dt)
    N = len(freqs) // 2
    x = np.fft.fftshift(freqs / f)[N:]
    snr = []
    _, ax = initialize_figure(figsize=(12,6))
    for i in range(len(fieldAml_list)):
        std = m.std()
        tmp = max(sf[i])/std
        snr.append(20*np.log10(abs(np.where(std == 0, 0, tmp))))
        ax.plot(x, sf[i, N:], color=color_list[i], marker='D', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(2, 12)
    ax.set_xticks(range(3, 12, 2))  # Set x-ticks every 2 units
    ax.set_ylabel(r'Magnitude(V/m)', weight='bold', fontsize=30)
    ax.set_xlabel('Harmonics Number', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'FieldEffect/figures/{figName}_Harmonics.png')
    #print snr
    print(f'SNR for {figName} nm:')
    for i in range(len(fieldAml_list)):
        print(f'SNR for field ampl {fieldAml_list[i]} mT = ', snr[i])
    print(50*'-')

def plot_PSF(xiH, sigH, m, dt, lent, cycs, color_list, trsp_list, figName):
    t = np.array([i*dt for i in range(lent)])
    _, ax = initialize_figure(figsize=(12,6))
    for i in range(len(fieldAml_list)):
        dm = fftDif(t, m[i])
        dxi = fftDif(t, xiH[i]+sigH[i])
        winlen = int(sigH.shape[1]/(2*cycs))
        dmdxi = dm/dxi
        wincnt = 2*cycs
        for j in range(wincnt):
            tmp = dmdxi[j*winlen : (j+1)*winlen]
            dmdxi[j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen, 20)*tmp, 10, 1, mode='nearest')
        ax.plot(xiH[i, 2*winlen:4*winlen]*1e3, dmdxi[2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0) 
    ax.set_ylabel(r'dm/d$\xi$', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\xi^{\prime}$', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'FieldEffect/figures/{figName}_psf.png')

def plot_FWHM(xiH, sigH, m, period, figName):

    leftxiH, leftpsf, rightxiH, rightpsf = psf_xiH(xiH[2], sigH[i], m[2], dt, lent, winlen, period)
    resl = fwhm_and_psf_peaks(leftxiH, leftpsf)
    resr = fwhm_and_psf_peaks(rightxiH, rightpsf)

    _, ax = initialize_figure(figsize=(12, 6))
    ax.plot(leftxiH * 1e3,  leftpsf, linewidth=3.0)
    ax.plot(resl['peakxiH'] * 1e3, resl['peakpsf'], 'rp', markersize=10, linewidth=3.0)
    ax.axvspan(resl['fwhm_left'] * 1e3, resl['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)
    ax.plot(rightxiH * 1e3, rightpsf, linewidth=3.0)
    ax.plot(resr['peakxiH'] * 1e3, resr['peakpsf'], 'rp', markersize=10, linewidth=3.0)
    ax.axvspan(resl['fwhm_left'] * 1e3, resr['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)
    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'FieldEffect/figures/{figName}_fwhm.png')

def plot_snr_fwhm(fieldAmpl, snr, fwhm, figName):
    fig, ax1 = initialize_figure(figsize=(12,6))

    # Plotting SNR on the left y-axis
    ax1.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax1.set_ylabel('SNR(dB)', color='tab:blue', weight='bold', fontsize=30)
    ax1.plot(fieldAmpl, snr, marker='o', color='tab:blue', markersize = 15, linewidth=3.0)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    set_spines_grid(ax1)
    # Plotting FWHM on the right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('FWHM', color='tab:red', weight='bold', fontsize=30)
    ax2.plot(fieldAmpl, fwhm, marker='s', color='tab:red', markersize = 15, linewidth=3.0)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    set_spines_grid(ax2)
    # Titles and grid
    fig.tight_layout()
    plt.savefig(f'FieldEffect/figures/{figName}_snr_fwhm.png')

def print_fwhm(xiH, sigH, m, dt, lent, winlen, figName):
    print(f'fwhm for {figName} nm:')
    for i, field in enumerate(fieldAml_list):  
        res = fwhm(xiH[i], sigH[i], m[i], dt, lent, winlen, 2)
        print(f'fwhm for field ampl {field} mT = ', res)
    print(50*'-')


if __name__ == '__main__':
    print(50*'-')
    data = Data() 
    f = data.fieldFreq
    B = data.fieldAmpl
    ka = data.kAnis
    Ms = data.Ms
    num = data.nParticle
    kT = data.kB*data.temp
    cycs = data.nPeriod
    fieldAml_list = np.array([20, 15, 10, 5])
    rsol_list = np.array([15000, 15000, 15000])

    dco_list = np.array([25, 30, 35])
    figName_list = ["Core25", "Core30", "Core35"]
    colorName_list = ['forest', 'sunset', 'red-brown', 'grapes']
    colorVar_list = [['lime', 'seagreen', 'forestgreen'], \
                    ['gold', 'orange', 'darkorange'], \
                    ['lightcoral', 'red', 'firebrick'], \
                    ['violet', 'fuchsia', 'mediumvioletred']]

    for i  in range(len(dco_list)):
        # read data
        figName = figName_list[i]
        dco = dco_list[i]*1e-9
        dhyd = dco+10e-9
        data.dCore = dco
        data.dHyd = dhyd
        data.rsol = rsol_list[i]
        params = MPI_Langevin_std_init(data)
        fs = params.fs
        print("original sampling frequency fs = ", fs)
        dwnSampFac = 200
        fsub = fs/dwnSampFac
        print("target sampling frequency fsub = ", fsub)
        #dt = params.dt # without subsampling
        dt = 1/fsub
        tf = params.tf
        lent = int(np.ceil(tf/dt))
        winlen = int(lent/(2*cycs))
        t = np.array([i*dt for i in range(lent)])
        He = np.zeros((len(fieldAml_list), lent))
        for i, a in enumerate(fieldAml_list):
            He[i,:] = np.array([a*1e-3 * Ht(f,i*dt) for i in range(lent)])

        color = colorName_list[i]
        colorVar = colorVar_list[i]
        mraw = genfromtxt(f'FieldEffect/data/{figName}_M.csv', delimiter=',')
        nraw = genfromtxt(f'FieldEffect/data/{figName}_N.csv', delimiter=',')
        sigHraw = genfromtxt(f'FieldEffect/data/{figName}_sigH.csv', delimiter=',')

        m = np.zeros((mraw.shape[0], lent))
        n = np.zeros((nraw.shape[0], lent))
        sigH = np.zeros((sigHraw.shape[0], lent))
        for i in range(len(mraw)):
            m[i] = subsample_signal(mraw[i], params.fs, fsub, dwnSampFac)
            n[i] = subsample_signal(nraw[i], params.fs, fsub, dwnSampFac)
            sigH[i] = subsample_signal(sigHraw[i], fs, fsub, dwnSampFac)

        color_list, trsp_list = colorMap(fieldAml_list, color, colorVar)

        # M_t
        plot_M_t(t, He, m, Ms, color_list, trsp_list, figName)

        # M_H
        plot_M_H(He, m, Ms, winlen, color_list, trsp_list, figName)

        # signals and Harmonics
        Vc = 1 / 6 * np.pi * dco ** 3
        sig = ka*Vc/kT
        gz = 1 # 1T = * 795.7747 A/m
        pz = 20e-3 # 1mT = 795.7747 A/m, I = 1.59 A
        mu = Ms * Vc
        xiH = np.zeros((len(fieldAml_list), lent))
        xi0 = np.zeros(len(fieldAml_list))
        lz = 1e-3
        for j, a in enumerate(fieldAml_list):
            xi0[j] = mu * a*1e-3 / (2*kT)
            xiH[j,:] = np.array([xi0[j] * Ht(f, k*dt) for k in range(lent)])
        plot_Signal(xiH, sigH, m, lz, dt, f, num, mu, lent, pz, cycs, figName)

        # PSF
        plot_PSF(xiH, sigH, m, dt, lent, cycs, color_list, trsp_list, figName)

        # PSF + FWHM for test
        period = 2
        plot_FWHM(xiH, sigH, m, period, figName)

        # Plot for SNR 25 and FWHM 25
        sf = np.zeros((len(fieldAml_list), lent))
        for i in range(len(fieldAml_list)):
            _, sf[i] = ftsignal(xiH[i], sigH[i], m[i], lz, dt, mu, num, lent, pz, cycs)   
        snr_list = []
        fwhm_list = []
        for i in range(len(fieldAml_list)):
            std = m.std()
            tmp = max(sf[i])/std
            snr_list.append(20*np.log10(abs(np.where(std == 0, 0, tmp))))
            tmp = fwhm(xiH[i], sigH[i], m[i], dt, lent, winlen, 2)
            fwhm_list.append(tmp)
        plot_snr_fwhm(fieldAml_list, snr_list, fwhm_list, figName)
        
        # FWHM data
        print_fwhm(xiH, sigH, m, dt, lent, winlen, figName)

        # print Neel relaxation
        ka = data.kAnis
        gam = data.gamGyro
        al = data.alpha
        Vc = 1 / 6 * np.pi * dco ** 3
        sig = ka * Vc / kT
        t0 = mu / (2 * gam * kT) * (1 + al ** 2) / al
        print(f'Neel relaxation for {figName} s:')
        for i, field in enumerate(fieldAml_list):  
            tn, hac = NeelRelaxation(sig, t0, ka, Ms, xi0[i], field*1e-3)
            print(f'tn for field ampl {field} (s) = ', tn)
            print(f'hac for field ampl {field} = ', hac)
            print(f'sigma for field ampl {field} = ', sig)
        print(50*'-')



