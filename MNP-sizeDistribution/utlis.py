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

Ht = lambda f, t: np.cos(2*np.pi*f*t)

def CombinedNeelBrown(data):
    al = data.alpha
    gam = data.gamGyro
    visc = data.visc
    cycs = data.nPeriod
    num = data.nParticle
    kT = data.kB*data.temp
    B = data.fieldAmpl
    f = data.fieldFreq
    Ms = data.Ms
    ka = data.kAnis
    # size dependent parameters
    dco = np.random.lognormal(mean=np.log(data.meanCore), sigma=data.stdCore, size=(num,1))
    dhy = dco + data.coatingSize
    Vc = 1 / 6 * np.pi * dco ** 3
    Vh = 1 / 6 * np.pi * dhy ** 3
    mu = Ms * Vc
    sig = ka * Vc / kT
    t0 = mu / (2 * gam * kT) * (1 + al ** 2) / al
    tB = 3 * visc * Vh / kT
    xi0 = mu * B / kT
    fs = data.rsol*2*f 
    # sampling
    dt = 1/fs
    tf = cycs*(1/f)
    lent = int(np.ceil(tf/dt))
    wrf = 1e-3 # this is used to reduce the Wiener noise
    ut = wrf*dt/tB
    vt = wrf*dt/t0
    nrf = 1e-1 # this is the noise reduction factor

    M = np.zeros((lent, 3))
    N = np.zeros((lent, 3))
    m = np.tile([1, 0, 0], (num, 1))
    n = m.copy()
    xI = np.zeros((num,3))
    for i in range(num):
      xI[i,2]= xi0[i]

    for j in range(lent):
        M[j, 0] = np.average(m[:,0], axis=0, weights=Vc.reshape(-1))
        M[j, 1] = np.average(m[:,1], axis=0, weights=Vc.reshape(-1))
        M[j, 2] = np.average(m[:,2], axis=0, weights=Vc.reshape(-1))
        N[j, 0] = np.average(n[:,0], axis=0, weights=Vc.reshape(-1))
        N[j, 1] = np.average(n[:,1], axis=0, weights=Vc.reshape(-1))
        N[j, 2] = np.average(n[:,2], axis=0, weights=Vc.reshape(-1))
        # M[j, :] = np.mean(m, axis=0)
        # N[j, :] = np.mean(n, axis=0)

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

        xb = xI * Ht(f,(j+1)*dt) + 2*sig * a2[:, np.newaxis] * n

        h2 = np.random.randn(num, 3)
        f2 = np.cross(xb / al + np.cross(mb, xb), mb) / 2
        g2 = np.cross(h2 / al + np.cross(mb, h2), mb)

        m = m + (f1 + f2) * vt / 2 + (g1 + g2) * np.sqrt(vt) / 2
        m = m / np.linalg.norm(m, axis=1, keepdims=True)

        print('\r', 'time step in samples: ' + "." * 10 + " ", end=str(j)+'/'+str(lent-1))

    return M[:, -1], N[:, -1], np.sum(Vc.reshape(-1))
def peaks_analysis(He, dmdh, mask):
    peaksIdx, _ = find_peaks(dmdh)
    srt = np.sort(dmdh[peaksIdx])
    max_peak = srt[-1]
    idxMax = np.where(dmdh[peaksIdx] == max_peak)[0]
    spline = UnivariateSpline(mask, dmdh - max_peak / 2, s=0)
    roots = spline.roots()
    idxr1 = np.where(mask == int(np.ceil(roots[0])))[0]
    idxr2 = np.where(mask == int(roots[0]))[0]
    idxr3 = np.where(mask == int(np.ceil(roots[1])))[0]
    idxr4 = np.where(mask == int(roots[1]))[0]
    v1 = (He[idxr1] + He[idxr2])[0] / 2
    v2 = (He[idxr3] + He[idxr4])[0] / 2
    fwhm = abs(v2 - v1)
    res = {
        'dmdH_peak': max_peak,
        'He_peak': He[peaksIdx[idxMax]][0],
        'fwhm_left': v1,
        'fwhm_right': v2,
        'fwhm': fwhm,
    }
    return res