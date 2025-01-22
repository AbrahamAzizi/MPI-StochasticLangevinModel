from init import *
from utils import Ht
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
import seaborn as sns

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

def read_data(group_name):
    with h5py.File('MNP-sizeDistribution/data/data.h5', 'r') as f:
        group = f[group_name]
        init_data = group['init_data'].attrs
        B = init_data["fieldAmpl"]
        f = init_data["fieldFreq"]
        Ms = init_data["Ms"]
        cycs = init_data["nPeriod"]
        rsol = init_data["rsol"]
        Mz=[]
        xif=[]
        sigf=[]
        psf=[]
        uz=[]
        stdList=[]
        for std_key in [k for k in group.keys() if k.startswith('std_')]:
            subgroup = group[std_key]
            Mz.append(subgroup['Mz'][:])
            xif.append(subgroup['AC_field'][:])
            sigf.append(subgroup['anisotropy_field'][:])
            psf.append(subgroup['psf'][:])
            uz.append(subgroup['uz'][:])
            sv = subgroup.attrs['tcv']
            stdList.append(float(std_key.split('_')[1]))
            leftpeak_group = subgroup['peak_left']
            for key in leftpeak_group.keys():
                if key == "fwhm":
                    fwhml=leftpeak_group[key][()]
            rightpeak_group = subgroup['peak_right']
            for key in rightpeak_group.keys():
                if key == "fwhm":
                    fwhmr=rightpeak_group[key][()]
            print(f"{group_name}_fwhm for std_{std_key}: \n", max(fwhml, fwhmr))

    return Mz, xif, sigf, psf, uz, B, f, Ms, cycs, rsol, sv, stdList, fwhml, fwhmr

def plot_figures(group_name, name, colors):
    
    # data
    Mz, xif, sigf, psf, uz, B, f, Ms, cycs, rsol, sumv, stdList, fwhml, fwhmr = read_data(group_name)
    fs = rsol*2*f
    dt = 1/fs
    tf = cycs*(1/f)
    lent = int(np.ceil(tf/dt))
    t = np.array([i*dt for i in range(lent)])
    He = np.array([B*Ht(f,i*dt) for i in range(lent)])

    # magnetization-time
    color_list, trsp_list = colorMap(stdList, name, colors)
    fig, ax1 = initialize_figure()
    ax1.plot(t * 1e3, He * 1e3, 'black', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=30)
    ax1.set_ylabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    ax1.set_xlim(.01, .11)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    for i in range(len(stdList)):
        ax2.plot(t * 1e3, Ms*Mz[i] * 1e-3, color=color_list[i], alpha=trsp_list[i], linewidth=3.0, label=rf'$\sigma$= {stdList[i]}')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax2.xaxis.set_tick_params(labelsize=30)
    ax2.yaxis.set_tick_params(labelsize=30)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig(f'MNP-sizeDistribution/figures/{group_name}+"-magnetization-time".png')

    # Magnetization curve
    fig, ax = initialize_figure(figsize=(12,6))
    winlen = (np.where(He == np.min(He))[0][0] - np.where(He == np.max(He))[0][0])
    freqs = np.fft.fftfreq(lent, dt)
    N = len(freqs) // 2
    x = np.fft.fftshift(freqs / f)[N:]
    for i in range(len(stdList)):
        ax.plot(He[4*winlen:]* 1e3, Ms*Mz[i][4*winlen:] * 1e-3 , color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'MNP-sizeDistribution/figures/{group_name}-magnetization-curve.png')

    # harmonics
    fig, ax = initialize_figure(figsize=(16,6))
    for i in range(len(stdList)):
        uk = sumv[i]*np.fft.fft(uz[i])
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
    plt.savefig(f'MNP-sizeDistribution/figures/{group_name}-harmonics.png')

    # PSF
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(len(stdList)):
        winlen = (np.where(xif[i] == np.min(xif[i]))[0][0] - np.where(xif[i] == np.max(xif[i]))[0][0])
        ax.plot(xif[i][2*winlen:4*winlen], psf[i][2*winlen:4*winlen], color=color_list[i], alpha=trsp_list[i], linewidth=3.0)
    ax.set_ylabel(r'dm/d$\xi$', weight='bold', fontsize=30)
    ax.set_xlabel(r'$\xi^{\prime}$', weight='bold', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.set_xlim(-50, 50)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig(f'MNP-sizeDistribution/figures/{group_name}-psf.png')

if __name__ == '__main__':
    
    plot_figures("IPG30", 'forest', ['lime', 'seagreen', 'forestgreen'])
    plot_figures("SHS30", 'sunset', ['gold', 'orange', 'darkorange'])
    plot_figures("SHP25", 'red-brown', ['lightcoral', 'red', 'firebrick'])
    plot_figures("SHP15", 'grapes', ['violet', 'fuchsia', 'mediumvioletred'])