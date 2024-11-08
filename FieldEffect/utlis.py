import numpy as np
from init import *
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import seaborn as sns

def low_pass_filter(signal, order, fs, fpass):
  fnyq = 0.5 * fs        # Nyquist frequency
  cutoff = fpass / fnyq  # Normalized cutoff frequency
  b, a = butter(order, cutoff, btype='low')
  return filtfilt(b, a,  signal, padtype='even', padlen=100)

def moving_average_filter(signal, window_size):
    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, window, mode='same')
    return smoothed_signal

Ht = lambda f, t: np.cos(2*np.pi*f*t)

# Neel relaxation time Fannin and Charless
def NeelRelaxation(sig, t0):
  if sig < 1:
    return t0 * (1 - 2 / 5 * sig + 48 / 875 * sig ** 2) ** (-1)
  else:
    return t0 * np.exp(sig) / 2 * np.sqrt(np.pi / sig ** 3)

def CombinedNeelBrown(data):
    al = data.alpha
    gam = data.gamGyro
    visc = data.visc
    cycs = data.nPeriod
    num = data.nParticle
    kT = data.kB*data.temp
    B = data.fieldAmpl
    f = data.fieldFreq
    dco = data.dCore
    dhy = data.dHyd
    Ms = data.Ms
    ka = data.kAnis
    Vc = 1 / 6 * np.pi * dco ** 3
    Vh = 1 / 6 * np.pi * dhy ** 3
    mu = Ms * Vc
    sig = ka * Vc / kT
    t0 = mu / (2 * gam * kT) * (1 + al ** 2) / al
    tB = 3 * visc * Vh / kT
    xi0 = mu * B / kT
    fs = data.rsol*2*f 
    dt = 1/fs
    tf = cycs*(1/f)
    lent = int(np.ceil(tf/dt))
    wrf = 1e-3 # this is used to reduce the Wiener noise
    ut = wrf*dt/tB
    vt = wrf*dt/t0
    nrf = 1e-6 # this is the noise reduction factor

    M = np.zeros((lent, 3))
    N = np.zeros((lent, 3))
    m = np.tile([1, 0, 0], (num, 1))
    n = m.copy()
    xI= np.tile([0, 0, xi0], (num, 1))

    for j in range(lent):
        M[j, :] = np.mean(m, axis=0)
        N[j, :] = np.mean(n, axis=0)

        a = np.sum(m * n, axis=1)

        dn = sig*a[:, np.newaxis] * (m - a[:, np.newaxis] * n) * ut + nrf*np.cross(np.random.randn(num, 3), n) * np.sqrt(ut)
        n = n + dn
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        # total field over time
        xi = xI * Ht(f,j*dt) + 2*sig * a[:, np.newaxis] * n

        h = np.random.randn(num, 3)
        f1 = np.cross(xi / al + np.cross(m, xi), m) / 2
        g1 = np.cross(h / al + np.cross(m, h), m)
        mb = m + f1 * vt + nrf*g1 * np.sqrt(vt)

        a2 = np.sum(mb * n, axis=1)

        xb = xI * Ht(f,(j+1)*dt) * 2*sig * a2[:, np.newaxis] * n

        h2 = np.random.randn(num, 3)
        f2 = np.cross(xb / al + np.cross(mb, xb), mb) / 2
        g2 = np.cross(h2 / al + np.cross(mb, h2), mb)

        m = m + (f1 + f2) * vt / 2 + (g1 + g2) * np.sqrt(vt) / 2
        m = m / np.linalg.norm(m, axis=1, keepdims=True)

        print('\r', 'time step in samples: ' + "." * 10 + " ", end=str(j)+'/'+str(lent-1))

    return M[:, -1], N[:, -1]

def peaksInit(He, dMk, dH, cycs, H_range=(-18e-3, 18e-3)):
#    " This function takes the field He and dM/dH in a cutted bound of the field in which
#    the edge effects of the field is not present and return the values of H and dM/dH for left
#    and right peaks seperately "
    l = dMk.shape[0]
    khalf = int(l / (2 * cycs))   # half of a period to identify left peak
    Hl = He[-2*khalf: -khalf]     # the array values of H for left peak 
    dmdhl = dMk[-2*khalf: -khalf] / dH[-2*khalf: -khalf]    # dM/dH for the left peak
    Hr = He[-khalf:]    # the array values of H for the right peak
    dmdhr = dMk[-khalf:] / dH[-khalf:]  # dM/dH for the rigth peak

    maskl = np.where((Hl >= H_range[0]) & (Hl <= H_range[1]))[0]    # mask Hl to avoid edge field 
    maskr = np.where((Hr >= H_range[0]) & (Hr <= H_range[1]))[0]    # mask Hr to avoid edge field

    Hlmask = Hl[maskl]
    dMdHlmask = dmdhl[maskl]
    Hrmask = Hr[maskr]
    dMdHrmask = dmdhr[maskr]

    return Hlmask, dMdHlmask , maskl, Hrmask, dMdHrmask, maskr

def peaks_analysis(HeMask, dmdhMask, mask):

#  " This function gets the H and dM/dH with a specific range of the field (mask) and return
#  the peak value and index, the coordinates (value and index) for fwhm"

    peaksIdx, _ = find_peaks(dmdhMask)    # find peaks in dM/dH ( local max )
    srt = np.sort(dmdhMask[peaksIdx])     
    max_peak = srt[-1]                    # find the max peak ( global max )
    idxMax = np.where(dmdhMask[peaksIdx] == max_peak)[0]    # index of the peak
    spline = UnivariateSpline(mask, dmdhMask - max_peak / 2, s=0)   # root of an spline and dM/dH at half of the max ( peak ) 
    roots = spline.roots()
    # the root is real value to find index we get the two closest integer values and average
    idxr1 = np.where(mask == int(np.ceil(roots[0])))[0] 
    idxr2 = np.where(mask == int(roots[0]))[0]
    idxr3 = np.where(mask == int(np.ceil(roots[1])))[0]
    idxr4 = np.where(mask == int(roots[1]))[0]
    v1 = (HeMask[idxr1] + HeMask[idxr2])[0] / 2   # the value at the first root
    v2 = (HeMask[idxr3] + HeMask[idxr4])[0] / 2   # the value at the second root
    res = {
        'dmdH_peak': max_peak,
        'He_peak': HeMask[peaksIdx[idxMax]][0],
        'fwhm_left': v1,
        'fwhm_right': v2,
        'fwhm': abs(v2 - v1)    ## fwhm is the differences of these two values
    }
    return res
