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
  return filtfilt(b, a,  signal)

# Neel relaxation time Fannin and Charless
def NeelRelaxation(sig, t0):
  if sig < 1:
    return t0 * (1 - 2 / 5 * sig + 48 / 875 * sig ** 2) ** (-1)
  else:
    return t0 * np.exp(sig) / 2 * np.sqrt(np.pi / sig ** 3)

def CombinedNeelBrown(init_data):
    xi0 = init_data['unitless_energy']
    ut = init_data['normalized_brown_time_step']
    vt = init_data['normalized_neel_time_step']
    t0 = init_data['neel_event_time']
    tB = init_data['brown_relax_time']
    dt = init_data['time_step']
    Hthermal = init_data['field_noise_variance']
    Tthermal = init_data['torque_noise_variance']
    lent = init_data['evaluation_time_length']
    sig = init_data['unitless_anisotropy']
    al = init_data['constant_damping']
    num = init_data['number_of_particles']

    M = np.zeros((lent, 3))
    N = np.zeros((lent, 3))

    m = np.tile([1, 0, 0], (num, 1))
    n = m.copy()
    xI = .5*np.tile([0, 0, xi0], (num, 1))

    for j in range(lent):
        M[j, :] = np.mean(m, axis=0)
        N[j, :] = np.mean(n, axis=0)

        a = np.sum(m * n, axis=1)

        dn = (1/tB)*(sig*a[:, np.newaxis] * (m - a[:, np.newaxis] * n) * ut + np.cross(np.sqrt(Tthermal)*np.random.randn(num, 3), n) * np.sqrt(ut))
        n = n + dn
        n = n / np.linalg.norm(n, axis=1, keepdims=True)

        xi = xI * np.cos(2 * np.pi * j * dt) + 2*sig * a[:, np.newaxis] * n  # total field over time

        h = np.sqrt(Hthermal)*np.random.randn(num, 3)
        f1 = np.cross(xi / al + np.cross(m, xi), m) / 2
        g1 = np.cross(h / al + np.cross(m, h), m)
        mb = m + f1 * vt + g1 * np.sqrt(vt)

        a2 = np.sum(mb * n, axis=1)

        xb = xI * np.cos(2 * np.pi * (j + 1) * dt) + 2*sig * a2[:, np.newaxis] * n

        h2 = np.sqrt(Hthermal)*np.random.randn(num, 3)
        f2 = np.cross(xb / al + np.cross(mb, xb), mb) / 2
        g2 = np.cross(h2 / al + np.cross(mb, h2), mb)

        m = (1/t0)*(m + (f1 + f2) * vt / 2 + (g1 + g2) * np.sqrt(vt) / 2)
        m = m / np.linalg.norm(m, axis=1, keepdims=True)

        #print('\r', 'time steps: ' + "." * 10 + " ", end=str(j)+'/'+str(lent-1))

    return M[:,-1], N

def peaksInit(He, dMk, dH, cycs, H_range=(-15e-3, 15e-3)):
    l = dMk.shape[0]
    khalf = int(l / (2 * cycs))
    Hl = He[-2*khalf: -khalf]
    dmdhl = dMk[-2*khalf: -khalf] / dH[-2*khalf: -khalf]
    Hr = He[-khalf:]
    dmdhr = dMk[-khalf:] / dH[-khalf:]

    maskl = np.where((Hl >= H_range[0]) & (Hl <= H_range[1]))[0]
    maskr = np.where((Hr >= H_range[0]) & (Hr <= H_range[1]))[0]

    Hlmask = Hl[maskl]
    dMdHlmask = dmdhl[maskl]
    Hrmask = Hr[maskr]
    dMdHrmask = dmdhr[maskr]

    return Hlmask, dMdHlmask , maskl, Hrmask, dMdHrmask, maskr
def peaks_analysis(HeMask, dmdhMask, mask):
    peaksIdx, _ = find_peaks(dmdhMask)
    srt = np.sort(dmdhMask[peaksIdx])
    max_peak = srt[-1]
    idxMax = np.where(dmdhMask[peaksIdx] == max_peak)[0]
    spline = UnivariateSpline(mask, dmdhMask - max_peak / 2, s=0)
    roots = spline.roots()
    idxr1 = np.where(mask == int(np.ceil(roots[0])))[0]
    idxr2 = np.where(mask == int(roots[0]))[0]
    idxr3 = np.where(mask == int(np.ceil(roots[1])))[0]
    idxr4 = np.where(mask == int(roots[1]))[0]
    v1 = (HeMask[idxr1] + HeMask[idxr2])[0] / 2
    v2 = (HeMask[idxr3] + HeMask[idxr4])[0] / 2
    fwhm = abs(v2 - v1)
    res = {
        'dmdH_peak': max_peak,
        'He_peak': HeMask[peaksIdx[idxMax]][0],
        'fwhm_left': v1,
        'fwhm_right': v2
    }
    return res
