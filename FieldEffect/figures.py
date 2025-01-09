from init import *
from utlis import psf_xiH, fwhm_and_psf_peaks, Ht, fwhm, dxi_dt
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

def plot_M_H(He, m, Ms, color_list, trsp_list, figName):
    _, ax = initialize_figure(figsize=(12,6))
    for i in range(m.shape[0]):
        winlen = (np.where(He[i] == np.min(He[i]))[0][0] - np.where(He[i] == np.max(He[i]))[0][0])
        ax.plot(He[i, 2*winlen:4*winlen]*1e3, Ms*m[i, 2*winlen:4*winlen]*1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'FieldEffect/figures/{figName}_M_H.png')

def plot_Harmonics(t, st, color_list, trsp_list, figName):
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    N = len(freqs) // 2
    x = np.fft.fftshift(freqs / f)[N:]
    snr = []
    _, ax = initialize_figure(figsize=(12,6))
    for i in range(len(fieldAml_list)):
        uk = np.fft.fft(st[i])
        y = abs(np.fft.fftshift(uk)/len(uk))[N:]  # 1e6 for scaling to uv
        x_int = np.array([2*k+1 for k in range(1, 21)])
        y_int = [y[np.argmin(np.abs(x - j))] for j in x_int]
        sd = st[i].std()
        snr.append(20*np.log10(abs(np.where(sd == 0, 0, y[3]/sd))))
        ax.plot(x_int, y_int, color=color_list[i], marker='D', markersize = 15, alpha=trsp_list[i], linewidth=3.0)
    ax.set_xlim(2, 20)
    ax.set_xticks(range(3, 20, 2))  # Set x-ticks every 2 units
    ax.set_ylabel(r'Magnitude(V)', weight='bold', fontsize=30)
    ax.set_xlabel('Harmonics Number', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'FieldEffect/figures/{figName}_Harmonics.png')
    #print snr
    print(f'SNR for {figName} nm:')
    for i in range(len(m)):
        print(f'SNR for field ampl {fieldAml_list[i]} mT = ', snr[i])
    print(50*'-')

def plot_Signal_Harmonics(xiH, sigH, m, n, xi0, lz, sig, dt, f, lent, pz, figName):
    _, ax = initialize_figure(figsize=(12,6))
    st = np.zeros((len(fieldAml_list), lent-1))
    for i in range(len(m)):
        leftxiH, leftpsf, rightxiH, rightpsf = psf_xiH(xiH[i], sigH[i], m[i], 2)
        dxidt = dxi_dt(xi0[i], sig, dt, f, lent, m[i], n[i])
        tmp = np.convolve(rightpsf,dxidt, mode='same')
        st[i] = -(pz*mu*num/lz[i])*tmp
        ax.plot(t[:-1]*1e3,st[i], color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('V(v)', weight='bold', fontsize=30)
    ax.set_xlabel('t(ms)', weight='bold', fontsize=30)
    ax.set_xlim(.01, .11)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'FieldEffect/figures/{figName}_Signals.png')
    #plot_Harmonics
    plot_Harmonics(t, st, color_list, trsp_list, figName)

def plot_PSF(xiH, sigH, m, color_list, trsp_list, figName):
    _, ax = initialize_figure(figsize=(12,6))
    for i in range(len(fieldAml_list)):
        dm = np.diff(m[i])
        dxi = np.diff(xiH[i]+sigH[i])
        dmdxi = dm/dxi
        winlen = (np.where(xiH[i] == np.min(xiH[i]))[0][0] - np.where(xiH[i] == np.max(xiH[i]))[0][0])
        wincnt = int(len(xiH[i])/winlen)
        for j in range(wincnt):
            tmp = dmdxi[j*winlen : (j+1)*winlen]
            dmdxi[j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,14)*tmp, 25, 2, mode='nearest')
        ax.plot(xiH[i, 2*winlen:4*winlen]*1e3, dmdxi[2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0) 
    ax.set_ylabel(r'dm/d$\xi$', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\xi^{\prime}$', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'FieldEffect/figures/{figName}_psf.png')

def plot_FWHM(xiH, sigH, m, period, figName):

    leftxiH, leftpsf, rightxiH, rightpsf = psf_xiH(xiH[2], sigH[i], m[2], period)
    resl = fwhm_and_psf_peaks(leftxiH, leftpsf)
    resr = fwhm_and_psf_peaks(rightxiH, rightpsf)

    _, ax = initialize_figure(figsize=(12, 6))
    ax.plot(leftxiH[:-1] * 1e3,  leftpsf, linewidth=3.0)
    ax.plot(resl['peakxiH'] * 1e3, resl['peakpsf'], 'rp', markersize=10, linewidth=3.0)
    ax.axvspan(resl['fwhm_left'] * 1e3, resl['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)
    ax.plot(rightxiH[:-1] * 1e3, rightpsf, linewidth=3.0)
    ax.plot(resr['peakxiH'] * 1e3, resr['peakpsf'], 'rp', markersize=10, linewidth=3.0)
    ax.axvspan(resl['fwhm_left'] * 1e3, resr['fwhm_right'] * 1e3, facecolor='c', alpha=0.1, linewidth=3.0)
    ax.set_ylabel(r'dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'FieldEffect/figures/{figName}_fwhm.png')

def print_fwhm(xiH, sigH, m, figName):
    print(f'fwhm for {figName} nm:')
    for i, field in enumerate(fieldAml_list):  
        res = fwhm(xiH[i], sigH[i], m[i], 2)
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
    data.rsol = 100
    rsol = data.rsol
    fs = rsol*2*f
    dt = 1/fs
    tf = cycs*(1/f)
    lent = int(np.ceil(tf/dt))
    t = np.array([i*dt for i in range(lent)])

    # preparing fields
    fieldAml_list = np.array([20, 15, 10, 5])
    He = np.zeros((len(fieldAml_list), len(t)))
    for i, a in enumerate(fieldAml_list):
        He[i,:] = np.array([a*1e-3 * Ht(f,i*dt) for i in range(lent)])

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
        color = colorName_list[i]
        colorVar = colorVar_list[i]
        m = genfromtxt(f'FieldEffect/data/{figName}_M.csv', delimiter=',')
        n = genfromtxt(f'FieldEffect/data/{figName}_N.csv', delimiter=',')
        sigH = genfromtxt(f'FieldEffect/data/{figName}_sigH.csv', delimiter=',')

        color_list, trsp_list = colorMap(fieldAml_list, color, colorVar)

        # M_t
        plot_M_t(t, He, m, Ms, color_list, trsp_list, figName)

        # M_H
        plot_M_H(He, m, Ms, color_list, trsp_list, figName)

        # signals and Harmonics
        Vc = 1 / 6 * np.pi * dco ** 3
        sig = ka*Vc/kT
        gz = 1 # 1T = * 795.7747 A/m
        pz = 20e-3 * 795.7747 / 1.59  # A/m/A   1T = 795.7747 A/m, I = 1.59 A
        mu = Ms * Vc
        xiH = np.zeros((len(fieldAml_list), len(t)))
        xi0 = np.zeros(len(fieldAml_list))
        lz = np.zeros(len(fieldAml_list))
        for i, a in enumerate(fieldAml_list):
            lz[i] = a*1e-3/gz
            xi0[i] = mu * a*1e-3 / (2*kT)
            xiH[i,:] = np.array([xi0[i] * Ht(f,j*dt) for j in range(lent)])
        st = np.zeros((len(fieldAml_list), lent-1))
        st = plot_Signal_Harmonics(xiH, sigH, m, n, xi0, lz, sig, dt, f, lent, pz, figName)

        # PSF
        plot_PSF(xiH, sigH, m, color_list, trsp_list, figName)

        # PSF + FWHM for test
        period = 2
        plot_FWHM(xiH, sigH, m, period, figName)

        # FWHM data
        print_fwhm(xiH, sigH, m, figName)



