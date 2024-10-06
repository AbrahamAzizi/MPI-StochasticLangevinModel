from init import *
from utlis import low_pass_filter, peaksInit, peaks_analysis
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    params = Params( )

    f = params.f_excitation
    t = params.tu / f
    Ms = 3e5

    # Magnetization in time for core size 25 nm
    M = genfromtxt('CoreSize25.csv', delimiter=',')
    n, l = M.shape
    k = int(l / params.nPeriod)
    minDistList = np.array([35, 60, 80, 100, 150, 200, 300, 500])
    He= params.fieldB * np.cos(2 * np.pi * params.tu)

    fig, ax1 = initialize_figure()
    ax1.plot(t * 1e3, He * 1e3, '-b', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    for i in range(M.shape[0]):
        ax2.plot(t * 1e3, Ms * M[i, :] * 1e-3, linewidth=3.0, label=f'$dist$= {minDistList[i]} nm')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('CoreSize25-time.png')

    # Magnetization curve for core size 25 nm
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        ax.plot(He[-k:] * 1e3, Ms * M[i, -k:] * 1e-3, linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('CoreSize15-magnetization-curve.png')

    # psf for core size 15 nm
    M_lowPass=np.zeros(M.shape)
    for i in range(M.shape[0]):
        M_lowPass[i,:] = low_pass_filter(M[i,:], 1, 1000*f, 3*f)
    dM = np.diff(np.append(M_lowPass, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, 0))
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[-k:]*1e3, dM[i, -k:]/dH[-k:] ,linewidth=3.0)
    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(-18,18)
    ax.set_ylim(0, 700)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('CoreSize15-psf.png')

    # Magnetization in time for core size 20 nm
    M = genfromtxt('CoreSize20.csv', delimiter=',')
    n, l = M.shape
    k = int(l / params.nPeriod)
    minDistList = np.array([30,50,80,100,150,200,300,500])
    He= params.fieldB * np.cos(2 * np.pi * params.tu)

    fig, ax1 = initialize_figure()
    ax1.plot(t * 1e3, He * 1e3, '-b', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    for i in range(M.shape[0]):
        ax2.plot(t * 1e3, Ms * M[i, :] * 1e-3, linewidth=3.0, label=f'$dist$= {minDistList[i]} nm')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('CoreSize20-time.png')

    # Magnetization curve for core size 20 nm
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        ax.plot(He[-k:] * 1e3, Ms * M[i, -k:] * 1e-3, linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('CoreSize20-magnetization-curve.png')

    # psf for core size 20 nm
    M_lowPass=np.zeros(M.shape)
    for i in range(M.shape[0]):
        M_lowPass[i,:] = low_pass_filter(M[i,:], 1, 1000*f, 3*f)
    dM = np.diff(np.append(M_lowPass, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, 0))
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[-k:]*1e3, dM[i, -k:]/dH[-k:] ,linewidth=3.0)
    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(-18,18)
    ax.set_ylim(0, 700)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('CoreSize20-psf.png')

    # Magnetization in time for core size 30 nm
    M = genfromtxt('CoreSize30.csv', delimiter=',')
    n, l = M.shape
    k = int(l / params.nPeriod)
    minDistList = np.array([40,50,80,100,150,200,300,500])
    He= params.fieldB * np.cos(2 * np.pi * params.tu)

    fig, ax1 = initialize_figure()
    ax1.plot(t * 1e3, He * 1e3, '-b', label='H', linewidth=3.0)
    ax1.set_xlabel('Time (ms)', weight='bold', fontsize=20)
    ax1.set_ylabel('$\mu_0$H (mT)', weight='bold', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax1)
    ax2 = ax1.twinx()
    for i in range(M.shape[0]):
        ax2.plot(t * 1e3, Ms * M[i, :] * 1e-3, linewidth=3.0, label=f'$dist$= {minDistList[i]} nm')
    ax2.set_ylabel('Mz (kA/m)', weight='bold', fontsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    set_spines_grid(ax2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.12, 1))
    set_legend_properties(legend)
    plt.tight_layout()
    plt.savefig('CoreSize30-time.png')

    # Magnetization curve for core size 30 nm
    fig, ax = initialize_figure(figsize=(12, 6))
    for i in range(M.shape[0]):
        ax.plot(He[:k] * 1e3, Ms * M[i, :k] * 1e-3, linewidth=3.0)
    ax.set_ylabel('Mz (kA/m)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('CoreSize30-magnetization-curve.png')

    # psf for core size 30 nm
    M_lowPass=np.zeros(M.shape)
    for i in range(M.shape[0]):
        M_lowPass[i,:] = low_pass_filter(M[i,:], 1, 1000*f, 3*f)
    dM = np.diff(np.append(M_lowPass, np.zeros((M.shape[0], 1)), axis=1), axis=1)
    dH = np.diff(np.append(He, 0))
    fig, ax = initialize_figure(figsize=(12,6))
    for i in range(M.shape[0]):
        ax.plot(He[:k]*1e3, dM[i, :k]/dH[:k] ,linewidth=3.0)
    ax.set_ylabel('dM/dH (A/m/$\mu_0$H)', weight='bold', fontsize=30)
    ax.set_xlabel('$\mu_0$H (mT)', weight='bold', fontsize=30)
    ax.set_xlim(-18,18)
    ax.set_ylim(0, 700)
    set_spines_grid(ax)
    plt.tight_layout()
    plt.savefig('CoreSize30-psf.png')






