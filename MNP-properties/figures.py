from init import *
from utils import psf_xiH, fwhm_and_psf_peaks, Ht, fwhm, ftSignal, low_pass_filter, fourier_derivative
import numpy as np
from numpy import genfromtxt
from scipy.signal import savgol_filter
from scipy.signal.windows import kaiser
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


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

def plot_M_t(t, He, m, Ms, color_list, trsp_list, proplist, figName):
    _, ax1 = initialize_figure()
    ax1.plot(t * 1e3, He * 1e3, color='b', linewidth=3.0, label=r'$\mu_0 H$ (mT)')
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
        ax2.plot(t * 1e3, Ms*m[i, :] * 1e-3 , '--', color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=f'{figName}: {proplist[i]} nm')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.set_xlim(.01, .11)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig(f'MNP-properties/figures/{figName}_M_t.png')

def plot_M_H(He, m, Ms, color_list, trsp_list, figName):
    _, ax = initialize_figure(figsize=(12,6))
    winlen = (np.where(He == np.min(He))[0][0] - np.where(He == np.max(He))[0][0])
    for i in range(len(m)):
        ax.plot(He[2*winlen:4*winlen]*1e3, Ms*m[i, 2*winlen:4*winlen]*1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'MNP-properties/figures/{figName}_M_H.png')

def plot_Signal(xiH, sigHlow, mlow, nlow, xi0, lz, sig, dt, f, lent, pz, num, mu, cycs, proplist, figName):
    _, ax = initialize_figure(figsize=(12,6))
    sf = np.zeros((len(proplist), lent-1))
    for i in range(len(proplist)):
        st, sf[i] = ftSignal(xiH[i], sigHlow[i], mlow[i], nlow[i], xi0[i], lz, sig, dt, f, lent, pz, num, mu[i], cycs)
        ax.plot(t[:-1]*1e3,st, color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('V/m', weight='bold', fontsize=30)
    ax.set_xlabel('t(ms)', weight='bold', fontsize=30)
    ax.set_xlim(.02, .12)
    #ax.set_ylim(-2.5e-5, 2.5e-5)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'MNP-properties/figures/{figName}_Signals.png')
    #plot_Harmonics
    freqs = np.fft.fftfreq(lent-1, dt)
    N = len(freqs) // 2
    x = np.fft.fftshift(freqs / f)[N:]
    snr = []
    _, ax = initialize_figure(figsize=(12,6))
    for i in range(len(proplist)):
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
    plt.savefig(f'MNP-properties/figures/{figName}_Harmonics.png')
    #print snr
    print(f'SNR for {figName}:')
    for i in range(len(proplist)):
        print(f'SNR for {figName} {proplist[i]} = ', snr[i])
    print(50*'-')

def plot_PSF(xiH, sigHlow, mlow, proplist, dt, color_list, trsp_list, figName):
    _, ax = initialize_figure(figsize=(12,6))
    for i in range(len(proplist)):
        dm = fourier_derivative(mlow[i], dt)
        dxi = fourier_derivative((xiH[i]+sigHlow[i]), dt)
        dmdxi = dm/dxi
        winlen = (np.where(xiH[i] == np.min(xiH[i]))[0][0] - np.where(xiH[i] == np.max(xiH[i]))[0][0])
        wincnt = int(len(xiH[i])/winlen)
        for j in range(wincnt):
            tmp = dmdxi[j*winlen : (j+1)*winlen]
            dmdxi[j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen,14)*tmp, 500, 1, mode='nearest')
        ax.plot(xiH[i, 2*winlen:4*winlen]*1e3, dmdxi[2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0) 
    ax.set_ylabel(r'dm/d$\xi$', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\xi^{\prime}$', weight='bold', fontsize=30)
    #ax.set_xlim(-5000, 5000)
    ax.set_ylim(0, )
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'MNP-properties/figures/{figName}_psf.png')

def plot_FWHM(xiH, sigHlow, mlow, period, figName):

    leftxiH, leftpsf, rightxiH, rightpsf = psf_xiH(xiH[-2], sigHlow[-2], mlow[-2], period)
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
    plt.savefig(f'MNP-properties/figures/{figName}_fwhm.png')

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
    plt.savefig(f'MNP-properties/figures/{figName}_snr_fwhm.png')

def print_fwhm(xiH, sigHlow, mlow, proplist, figName):
    print(f'fwhm for {figName}:')
    for i, field in enumerate(proplist):  
        res = fwhm(xiH[i], sigHlow[i], mlow[i], 2)
        print(f'fwhm for {figName} {field} = ', res)
    print(50*'-')


if __name__ == '__main__':
    print(50*'-')
    data = Data()
    params = Params(data)
    
    freq = data.fieldFreq
    cycs = data.nPeriod
    num = data.nParticle
    lent = params.lent
    t = params.tu/freq
    He = data.fieldAmpl*np.cos(2*np.pi*freq*t)

    # Core size --------------------------------------------------
    Ms = data.Ms
    dco_list = np.array([20, 30, 40, 50, 60])
    color_list, trsp_list = colorMap(dco_list, 'forest', ['lime', 'seagreen', 'forestgreen'])
    m = genfromtxt('MNP-properties/data/Core_M.csv', delimiter=',')
    n = genfromtxt('MNP-properties/data/Core_N.csv', delimiter=',')
    sigH = genfromtxt('MNP-properties/data/Core_sigH.csv', delimiter=',')
    mlow = np.zeros((len(m), lent))
    nlow = np.zeros((len(n), lent))
    sigHlow = np.zeros((len(sigH), lent))
    for i in range(len(m)):
        mlow[i]= low_pass_filter(m[i], 4, 400*freq, 5*freq)
        nlow[i]= low_pass_filter(n[i], 4, 400*freq, 5*freq)
        sigHlow[i]= low_pass_filter(sigH[i], 4, 400*freq, 5*freq)

    # M_t
    plot_M_t(t, He, mlow, Ms, color_list, trsp_list, dco_list, "Core Size")

    # M_H
    plot_M_H(He, mlow, Ms, color_list, trsp_list, "Core Size")

    # signals and Harmonics
    gz = 1 # 1T = * 795.7747 A/m
    pz = 20e-3 # 1mT = 795.7747 A/m, I = 1.59 A
    dt = t[2]-t[1]
    xiH = np.zeros((len(dco_list), lent))
    xi0 = np.zeros(len(dco_list))
    mulist = np.zeros(len(dco_list))
    lz = 1e-3
    for j, d in enumerate(dco_list):
        data.dCore = d*1e-9
        Params(data)
        sig = params.sig
        mulist[j] = params.mu
        xi0[j] = params.xi0
        xiH[j,:] = xi0[j] * Ht(freq, t)

    plot_Signal(xiH, sigHlow, mlow, nlow, xi0, lz, sig, dt, freq, lent, pz, num, mulist, cycs, dco_list, "Core Size")
    
    # PSF
    plot_PSF(xiH, sigHlow, mlow, dco_list, dt, color_list, trsp_list, "Core Size")

    # PSF + FWHM for test
    period = 2
    plot_FWHM(xiH, sigHlow, mlow, period, "Core Size")

    # Plot SNR and FWHM
    sf = np.zeros((len(dco_list), lent-1))
    for i in range(len(dco_list)):
        _, sf[i] = ftSignal(xiH[i], sigHlow[i], mlow[i], nlow[i], xi0[i], lz, sig, dt, freq, lent, pz, num, mulist[i], cycs)   
    snr_list = []
    fwhm_list = []
    for i in range(len(dco_list)):
        std = m.std()
        tmp = max(sf[i])/std
        snr_list.append(20*np.log10(abs(np.where(std == 0, 0, tmp))))
        tmp = fwhm(xiH[i], sigHlow[i], mlow[i], 2)
        fwhm_list.append(tmp)
    plot_snr_fwhm(dco_list, snr_list, fwhm_list, "Core Size")
    
    # FWHM data
    print_fwhm(xiH, sigHlow, mlow, dco_list, "Core Size")

    # Hyd size --------------------------------------------------
    Ms = data.Ms
    dhyd_list = np.array([30, 35, 40, 45, 55, 60])
    color_list, trsp_list = colorMap(dhyd_list, 'sunset', ['gold', 'orange', 'darkorange'])
    m = genfromtxt('MNP-properties/data/Hyd_M.csv', delimiter=',')
    n = genfromtxt('MNP-properties/data/Hyd_N.csv', delimiter=',')
    sigH = genfromtxt('MNP-properties/data/Hyd_sigH.csv', delimiter=',')
    mlow = np.zeros((len(m), lent))
    nlow = np.zeros((len(n), lent))
    sigHlow = np.zeros((len(sigH), lent))
    for i in range(len(m)):
        mlow[i]= low_pass_filter(m[i], 4, 400*freq, 5*freq)
        nlow[i]= low_pass_filter(n[i], 4, 400*freq, 5*freq)
        sigHlow[i]= low_pass_filter(sigH[i], 4, 400*freq, 5*freq)

    # M_t
    plot_M_t(t, He, mlow, Ms, color_list, trsp_list, dhyd_list, "Hyd Size")

    # M_H
    plot_M_H(He, mlow, Ms, color_list, trsp_list, "Hyd Size")

    # signals and Harmonics
    gz = 1 # 1T = * 795.7747 A/m
    pz = 20e-3 # 1mT = 795.7747 A/m, I = 1.59 A
    xiH = np.zeros((len(dhyd_list), lent))
    xi0 = np.zeros(len(dhyd_list))
    mulist = np.zeros(len(dhyd_list))
    lz = 1e-3
    for j, d in enumerate(dhyd_list):
        data.dHyd = d*1e-9
        Params(data)
        sig = params.sig
        mulist[j] = params.mu
        xi0[j] = params.xi0
        xiH[j,:] = xi0[j] * Ht(freq, t)

    plot_Signal(xiH, sigHlow, mlow, nlow, xi0, lz, sig, dt, freq, lent, pz, num, mulist, cycs, dco_list, "Hyd Size")
    
    # PSF
    plot_PSF(xiH, sigHlow, mlow, dco_list, dt, color_list, trsp_list, "Hyd Size")

    # PSF + FWHM for test
    period = 2
    plot_FWHM(xiH, sigHlow, mlow, period, "Hyd Size")

    # Plot SNR and FWHM
    sf = np.zeros((len(dhyd_list), lent-1))
    for i in range(len(dhyd_list)):
        _, sf[i] = ftSignal(xiH[i], sigHlow[i], mlow[i], nlow[i], xi0[i], lz, sig, dt, freq, lent, pz, num, mulist[i], cycs)   
    snr_list = []
    fwhm_list = []
    for i in range(len(dhyd_list)):
        std = m.std()
        tmp = max(sf[i])/std
        snr_list.append(20*np.log10(abs(np.where(std == 0, 0, tmp))))
        tmp = fwhm(xiH[i], sigHlow[i], mlow[i], 2)
        fwhm_list.append(tmp)
    plot_snr_fwhm(dhyd_list, snr_list, fwhm_list, "Hyd Size")
    
    # FWHM data
    print_fwhm(xiH, sigHlow, mlow, dhyd_list, "Hyd Size")

    # Kanis --------------------------------------------------
    Ms = data.Ms
    ka_list = np.array([3, 5, 8, 10, 15])
    color_list, trsp_list = colorMap(ka_list, 'grapes', ['violet', 'fuchsia', 'mediumvioletred'])
    m = genfromtxt('MNP-properties/data/Anis_M.csv', delimiter=',')
    n = genfromtxt('MNP-properties/data/Anis_N.csv', delimiter=',')
    sigH = genfromtxt('MNP-properties/data/Anis_sigH.csv', delimiter=',')
    mlow = np.zeros((len(m), lent))
    nlow = np.zeros((len(n), lent))
    sigHlow = np.zeros((len(sigH), lent))
    for i in range(len(m)):
        mlow[i]= low_pass_filter(m[i], 4, 400*freq, 5*freq)
        nlow[i]= low_pass_filter(n[i], 4, 400*freq, 5*freq)
        sigHlow[i]= low_pass_filter(sigH[i], 4, 400*freq, 5*freq)

    # M_t
    plot_M_t(t, He, mlow, Ms, color_list, trsp_list, ka_list, "Ka")

    # M_H
    plot_M_H(He, mlow, Ms, color_list, trsp_list, "Ka")

    # signals and Harmonics
    gz = 1 # 1T = * 795.7747 A/m
    pz = 20e-3 # 1mT = 795.7747 A/m, I = 1.59 A
    xiH = np.zeros((len(ka_list), lent))
    xi0 = np.zeros(len(ka_list))
    mulist = np.zeros(len(ka_list))
    lz = 1e-3
    for j, d in enumerate(ka_list):
        data.kAnis = d*1e-9
        Params(data)
        sig = params.sig
        mulist[j] = params.mu
        xi0[j] = params.xi0
        xiH[j,:] = xi0[j] * Ht(freq, t)

    plot_Signal(xiH, sigHlow, mlow, nlow, xi0, lz, sig, dt, freq, lent, pz, num, mulist, cycs, ka_list, "Ka")
    
    # PSF
    plot_PSF(xiH, sigHlow, mlow, dco_list, dt, color_list, trsp_list, "Ka")

    # PSF + FWHM for test
    period = 2
    plot_FWHM(xiH, sigHlow, mlow, period, "Ka")

    # Plot SNR and FWHM
    sf = np.zeros((len(ka_list), lent-1))
    for i in range(len(ka_list)):
        _, sf[i] = ftSignal(xiH[i], sigHlow[i], mlow[i], nlow[i], xi0[i], lz, sig, dt, freq, lent, pz, num, mulist[i], cycs)   
    snr_list = []
    fwhm_list = []
    for i in range(len(ka_list)):
        std = m.std()
        tmp = max(sf[i])/std
        snr_list.append(20*np.log10(abs(np.where(std == 0, 0, tmp))))
        tmp = fwhm(xiH[i], sigH[i], mlow[i], 2)
        fwhm_list.append(tmp)
    plot_snr_fwhm(ka_list, snr_list, fwhm_list, "Ka")
    
    # FWHM data
    print_fwhm(xiH, sigHlow, mlow, ka_list, "Ka")

    # Magnetization Saturation --------------------------------------------------
    ms_list = np.array([50, 100, 200, 300, 400, 500])
    color_list, trsp_list = colorMap(ms_list, 'red-brown', ['lightcoral', 'red', 'firebrick'])
    m = genfromtxt('MNP-properties/data/Satu_M.csv', delimiter=',')
    n = genfromtxt('MNP-properties/data/Satu_N.csv', delimiter=',')
    sigH = genfromtxt('MNP-properties/data/Satu_sigH.csv', delimiter=',')
    mlow = np.zeros((len(m), lent))
    nlow = np.zeros((len(n), lent))
    sigHlow = np.zeros((len(sigH), lent))
    for i in range(len(m)):
        mlow[i]= low_pass_filter(m[i], 4, 400*freq, 5*freq)
        nlow[i]= low_pass_filter(n[i], 4, 400*freq, 5*freq)
        sigHlow[i]= low_pass_filter(sigH[i], 4, 400*freq, 5*freq)

    # M_t
    plot_M_t(t, He, mlow, Ms, color_list, trsp_list, ms_list, "Ms")

    # M_H
    plot_M_H(He, mlow, Ms, color_list, trsp_list, "Ms")

    # signals and Harmonics
    gz = 1 # 1T = * 795.7747 A/m
    pz = 20e-3 # 1mT = 795.7747 A/m, I = 1.59 A
    xiH = np.zeros((len(ms_list), lent))
    xi0 = np.zeros(len(ms_list))
    mulist = np.zeros(len(ms_list))
    lz = 1e-3
    for j, d in enumerate(ms_list):
        data.Ms = d*1e3
        Params(data)
        sig = params.sig
        mulist[j] = params.mu
        xi0[j] = params.xi0
        xiH[j,:] = xi0[j] * Ht(freq, t)

    plot_Signal(xiH, sigHlow, mlow, nlow, xi0, lz, sig, dt, freq, lent, pz, num, mulist, cycs, ms_list, "Ms")
    
    # PSF
    plot_PSF(xiH, sigHlow, mlow, ms_list, dt, color_list, trsp_list, "Ms")

    # PSF + FWHM for test
    period = 2
    plot_FWHM(xiH, sigHlow, mlow, period, "Ms")

    # Plot SNR and FWHM
    sf = np.zeros((len(ms_list), lent-1))
    for i in range(len(ms_list)):
        _, sf[i] = ftSignal(xiH[i], sigHlow[i], mlow[i], nlow[i], xi0[i], lz, sig, dt, freq, lent, pz, num, mulist[i], cycs)   
    snr_list = []
    fwhm_list = []
    for i in range(len(ms_list)):
        std = m.std()
        tmp = max(sf[i])/std
        snr_list.append(20*np.log10(abs(np.where(std == 0, 0, tmp))))
        tmp = fwhm(xiH[i], sigHlow[i], mlow[i], 2)
        fwhm_list.append(tmp)
    plot_snr_fwhm(ms_list, snr_list, fwhm_list, "Ms")
    
    # FWHM data
    print_fwhm(xiH, sigHlow, mlow, ms_list, "Ms")