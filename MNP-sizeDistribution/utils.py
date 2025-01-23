import numpy as np
from init import *
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.signal.windows import kaiser

def low_pass_filter(signal, order, fs, fpass):
  fnyq = 0.5 * fs        # Nyquist frequency
  cutoff = fpass / fnyq  # Normalized cutoff frequency
  b, a = butter(order, cutoff, btype='low')
  return filtfilt(b, a,  signal)

Ht = lambda f, t: np.cos(2*np.pi*f*t)

def MPI_Langevin_std_init(data):
    return Params(data)

# for fwhm
def fwhm_core(xif, psf, mask):
    peaksIdx, _ = find_peaks(psf)
    srt = np.sort(psf[peaksIdx])
    max_peak = srt[-1]
    idxMax = np.where(psf[peaksIdx] == max_peak)[0]
    spline = UnivariateSpline(mask, psf - max_peak / 2, s=0)
    roots = spline.roots()
    idxr1 = np.where(mask == int(np.ceil(roots[0])))[0]
    idxr2 = np.where(mask == int(roots[0]))[0]
    idxr3 = np.where(mask == int(np.ceil(roots[1])))[0]
    idxr4 = np.where(mask == int(roots[1]))[0]
    v1 = (xif[idxr1] + xif[idxr2])[0] / 2
    v2 = (xif[idxr3] + xif[idxr4])[0] / 2
    res = {
        'dmdH_peak': max_peak,
        'He_peak': xif[peaksIdx[idxMax]][0],
        'fwhm_left': v1,
        'fwhm_right': v2,
        'fwhm': abs(v2 - v1),
    }
    return res

# fwhm and peaks data for left and right peaks
def fwhm_data(psf, xif):
  xifMax = np.max(xif)
  xifMin = np.min(xif)
  winlen = (np.where(xif == xifMin)[0][0] - np.where(xif == np.max(xif))[0][0])
  maskl = np.where((xif[2*winlen:3*winlen] >= xifMin) & (xif[2*winlen:3*winlen]<= xifMax))[0]
  resultl = fwhm_core(xif[2*winlen:3*winlen], psf[2*winlen:3*winlen], maskl)
  maskr = np.where((xif[3*winlen:4*winlen] >= xifMin) & (xif[3*winlen:4*winlen]<= xifMax))[0]
  resultr = fwhm_core(xif[3*winlen:4*winlen], psf[3*winlen:4*winlen], maskr)
  return resultl, resultr

def psf(m, xif, sigf):
  winlen = (np.where(xif == np.min(xif))[0][0] - np.where(xif == np.max(xif))[0][0])
  wincnt = int(len(xif)/winlen)
  dm = np.diff(m)
  xi = xif+sigf
  dxi = np.diff(xi)
  dmdH = dm/dxi 
  psfval = np.zeros(len(xif)-1)
  for j in range(wincnt):
    tmp = dmdH[j*winlen : (j+1)*winlen]
    psfval[j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen, 14)*tmp, 25, 2, mode='nearest')
  
  return psfval

def InducedVoltage(m, xif, pz, fs):
  u0 = 4 * np.pi * 1e-7  # T.m/A, v.s/A/m
  dt = 1/fs
  lent = len(xif)
  t = np.array([i*dt for i in range(lent)])
  winlen = (np.where(xif == np.min(xif))[0][0] - np.where(xif == np.max(xif))[0][0])
  wincnt = int(len(xif)/winlen)
  dm = np.diff(m)
  dift = np.diff(t)
  dmdt = -pz * u0 *dm/dift
  uz = np.zeros(lent-1)
  for j in range(wincnt):
    tmp = dmdt[j*winlen : (j+1)*winlen]
    uz[j*winlen : (j+1)*winlen] = savgol_filter(kaiser(winlen, 14)*tmp, 25, 2, mode='nearest')
  
  return uz

def MPI_Langevin_std_core(data):
  params = MPI_Langevin_std_init(data)
  al = data.alpha
  num = data.nParticle
  f = data.fieldFreq
  sig = params.sig
  xi0 = params.xi0
  Vc = params.Vc
  dt = params.dt
  lent = params.lent
  ut = params.ut
  vt = params.vt

  # in time
  M = np.zeros((lent, 3))           # average magnetization in time
  N = np.zeros((lent, 3))           # average easy axis in time
  xif_t = np.zeros((lent, 3))        # average unitless applied field (AC + gradient)
  sigf_t = np.zeros((lent, 3))      # average unitless anisotropy field

  # for particles
  m = np.tile([1, 0, 0], (num, 1))  # magnetization unit vector
  n = m.copy()                      # easy axis unit vector          
  xin = np.zeros((num,3))         # unitless amplitude applied field (AC + gradient) vector
  xif = np.zeros((num, 3))         # unitless applied field (AC + gradient) vector
  sigf = np.zeros((num, 3))         # unitless anisotropy field vector

  for i in range(num):
    xin[i,2]= xi0[i]

  for j in range(lent):
    M[j, 0] = np.average(m[:,0], axis=0, weights=Vc.reshape(-1))
    M[j, 1] = np.average(m[:,1], axis=0, weights=Vc.reshape(-1))
    M[j, 2] = np.average(m[:,2], axis=0, weights=Vc.reshape(-1))
    N[j, 0] = np.average(n[:,0], axis=0, weights=Vc.reshape(-1))
    N[j, 1] = np.average(n[:,1], axis=0, weights=Vc.reshape(-1))
    N[j, 2] = np.average(n[:,2], axis=0, weights=Vc.reshape(-1))
    xif_t[j, 0] = np.average(xif[:,0], axis=0, weights=Vc.reshape(-1))
    xif_t[j, 1] = np.average(xif[:,1], axis=0, weights=Vc.reshape(-1))
    xif_t[j, 2] = np.average(xif[:,2], axis=0, weights=Vc.reshape(-1))
    sigf_t[j, 0] = np.average(sigf[:,0], axis=0, weights=Vc.reshape(-1))
    sigf_t[j, 1] = np.average(sigf[:,0], axis=0, weights=Vc.reshape(-1))
    sigf_t[j, 2] = np.average(sigf[:,0], axis=0, weights=Vc.reshape(-1))

    a = np.sum(m * n, axis=1)
    dn = sig*a[:, np.newaxis] * (m - a[:, np.newaxis] * n) * ut + np.cross(np.random.randn(num, 3), n) * np.sqrt(ut)
    n = n + dn
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    # total field over time
    sigf = sig * a[:, np.newaxis] * n
    xif = xin * Ht(f,j*dt) 
    xi = xif + sigf
    
    h = np.random.randn(num, 3)
    f1 = np.cross(xi / al + np.cross(m, xi), m) 
    g1 = np.cross(h / al + np.cross(m, h), m)
    mb = m + f1 * vt + g1 * np.sqrt(vt)

    a2 = np.sum(mb * n, axis=1)
    xb = xin * Ht(f,(j+1)*dt) + sig * a2[:, np.newaxis] * n
    h2 = np.random.randn(num, 3)
    f2 = np.cross(xb / al + np.cross(mb, xb), mb) 
    g2 = np.cross(h2 / al + np.cross(mb, h2), mb)
    m = m + (f1 + f2) * vt / 2 + (g1 + g2) * np.sqrt(vt) / 2
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    print(f"\r{'time step in samples: ' + '.' * 10} {j}/{lent-1}", end="")
  return M, N, xif_t, sigf_t, np.sum(Vc.reshape(-1))