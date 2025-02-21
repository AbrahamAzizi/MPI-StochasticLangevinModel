import numpy as np
from init import *
from scipy.signal import butter, filtfilt, find_peaks, decimate
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.signal.windows import kaiser
import matplotlib.pyplot as plt
import seaborn as sns

def lowpass_filter(signal, order, fs, fpass):
  fnyq = 0.5 * fs        # Nyquist frequency
  cutoff = fpass / fnyq  # Normalized cutoff frequency
  b, a = butter(order, cutoff, btype='low')
  return filtfilt(b, a,  signal, padtype='even', padlen=100)

Ht = lambda f, t: np.cos(2*np.pi*f*t)

def fftDif(x, y):
  y_f = np.fft.fft(y)  # FFT of x
  freqs = np.fft.fftfreq(len(x), d=x[2]-x[1])  
  jw = 2j * np.pi * freqs  
  dy_f = jw * y_f  
  dy = np.fft.ifft(dy_f).real  
  return dy

def ftsignal(xiH, sigH, m, lz, dt, num, mu, lent, pz, cycs):
    """ This function return the signal in time and frequency """
    st = np.zeros(lent)
    sf = np.zeros(lent)
    t = np.array([i*dt for i in range(lent)])
    dmdt = fftDif(t, m)
    winlen = int(lent/(2*cycs))
    for i in range(cycs):
      j = 2*i
      _, leftpsf, _, rightpsf = psf_xiH(xiH, sigH, m, dt, lent, winlen, i+1)
      st[j*winlen:(j+1)*winlen] = -(pz*mu*num/lz)*(leftpsf)*dmdt[j*winlen:(j+1)*winlen]
      st[(j+1)*winlen:(j+2)*winlen] = -(pz*mu*num/lz)*(rightpsf)*dmdt[(j+1)*winlen:(j+2)*winlen]
    uk = np.fft.fft(st)
    sf = abs(np.fft.fftshift(uk))
    return st, sf

def subsample_signal(M, original_fs, target_fs):
    # Compute decimation factor
    decimation_factor = int(original_fs / target_fs)
    if original_fs % target_fs != 0:
        raise ValueError("Original sampling rate must be an integer multiple of the target sampling rate.")

    # Low-pass filter before decimation to avoid aliasing
    cutoff_frequency = target_fs / 2  # Nyquist frequency of the target sampling rate
    filtered_signal = lowpass_filter(M, 1, original_fs, cutoff_frequency)

    # Decimate the filtered signal
    subsignal = decimate(filtered_signal, decimation_factor, axis=-1, ftype='iir')
    return subsignal

# Neel relaxation time Fannin and Charless
def NeelRelaxation(sig, t0, ka, Ms, xi0, B):
  u0 = 4*np.pi*1e-7 
  Hac = B/u0
  Hk = 2*ka/(u0*Ms)
  hac = Hac/Hk
  a1 = .92 + .034*sig
  a2 = -.45 + .045*sig
  deltae = 1 - a1*hac + a2*hac**2
  if B <= 10e-3:
    tn = t0*np.sqrt(np.pi/sig)*(1/(1-hac))*np.exp(sig*deltae)
  elif B > 10e-3:
    tn1 = 0.5*t0*np.sqrt(np.pi/sig)*np.exp(sig)
    tn2 = np.sqrt(1+1.97*xi0**3.18)
    tn = tn1/tn2
  return tn, hac

def MPI_Langevin_std_init(data):
    return Params(data)

def MPI_Langevin_std_core(data):
    al = data.alpha
    cycs = data.nPeriod
    num = data.nParticle
    f = data.fieldFreq
    params = MPI_Langevin_std_init(data)
    sig = params.sig
    t0 = params.t0
    tB = params.tB
    xi0 = params.xi0
    fs = params.fs 
    dt = 1/fs
    tf = cycs*(1/f)
    lent = int(np.ceil(tf/dt))
    #wrf = 1e-3 # this is used to reduce the Wiener noise
    ut = dt/tB
    vt = dt/t0

    M = np.zeros((lent, 3))
    N = np.zeros((lent, 3))
    sigt = np.zeros((lent, 3))
    m = np.tile([1, 0, 0], (num, 1))
    n = m.copy()
    xI= np.tile([0, 0, xi0], (num, 1))
    sign = np.tile([0, 0, sig], (num, 1))

    for j in range(lent):
        M[j, :] = np.mean(m, axis=0)
        N[j, :] = np.mean(n, axis=0)
        sigt[j, :] = np.mean(sign, axis=0)

        a = np.sum(m * n, axis=1)

        dn = sig*a[:, np.newaxis] * (m - a[:, np.newaxis] * n) * ut + np.cross(np.random.randn(num, 3), n) * np.sqrt(ut)
        n = n + dn
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        # total field over time
        xi = xI * Ht(f,j*dt) + sig * a[:, np.newaxis] * n

        h = np.random.randn(num, 3)
        f1 = np.cross(xi / al + np.cross(m, xi), m)
        g1 = np.cross(h / al + np.cross(m, h), m)
        mb = m + f1 * vt + g1 * np.sqrt(vt)

        a = np.sum(mb * n, axis=1)
        xb = xI * Ht(f,(j+1)*dt) + sig * a[:, np.newaxis] * n

        h2 = np.random.randn(num, 3)
        f2 = np.cross(xb / al + np.cross(mb, xb), mb) 
        g2 = np.cross(h2 / al + np.cross(mb, h2), mb)

        m = m + (f1 + f2) * vt / 2 + (g1 + g2) * np.sqrt(vt) / 2
        m = m / np.linalg.norm(m, axis=1, keepdims=True)

        a = np.sum(m * n, axis=1)
        sign = sig * a[:, np.newaxis] * n

        print('\r', 'time step in samples: ' + "." * 10 + " ", end=str(j)+'/'+str(lent-1))

    return M[:, -1], N[:, -1], sigt[:, -1]

def psf_xiH(xiH, sigH, m, dt, lent, winlen, period):
  """ This function extract the psf and its related xiH values for a period 
  inputs:
    xiH: u*B/kT
    m: magnetization unite vecotr in direction of the applied field
    period: the desired period, T=1, T=2, T=3
  outputs:
    psf: dm/dxi
    leftxiH
    rightxiH """
  t = np.array([i*dt for i in range(lent)])
  leftsigH = sigH[(2*period-2)*winlen : (2*period-1)*winlen] 
  rightsigH = sigH[(2*period-1)*winlen : (2*period)*winlen]
  leftxiH = xiH[(2*period-2)*winlen : (2*period-1)*winlen] # the second period is used [2*winlen : 4*winlen]
  rightxiH = xiH[(2*period-1)*winlen : (2*period)*winlen]
  leftxi = leftxiH + leftsigH
  rightxi = rightxiH + rightsigH
  leftm = m[(2*period-2)*winlen : (2*period-1)*winlen]
  leftt = t[(2*period-2)*winlen : (2*period-1)*winlen]
  rightm = m[(2*period-1)*winlen : (2*period)*winlen]
  rightt = t[(2*period-1)*winlen : (2*period)*winlen]
  leftpsf = savgol_filter(kaiser(winlen, 14)*fftDif(leftt ,leftm)/fftDif(leftt, leftxi), 20, 1, mode='nearest')
  rightpsf = savgol_filter(kaiser(winlen, 14)*fftDif(rightt, rightm)/fftDif(rightt, rightxi), 20, 1, mode='nearest')
  return leftxiH, leftpsf, rightxiH, rightpsf

def fwhm_and_psf_peaks(xiH, psf):
  """ This function find peaks coordinates and fwhm data for each lobe 
  input:
    xiH: dimensionless magnetic field for one lobe
    psf: psf for one lobe
  output:
    a dictionary of peaks coordinates and fwhm of one lobe """
  peakpsf = max(psf)
  half_max = max(psf) / 2
  indices = np.where(psf >= half_max)[0]
  left_index = indices[0]
  right_index = indices[-1]
  v1 = xiH[left_index] 
  v2 = xiH[right_index] 
  fwhm = abs(v1 - v2)
  res = {
        'peakpsf': peakpsf,
        'peakxiH': xiH[np.where(psf==peakpsf)],
        'fwhm_left': v1,
        'fwhm_right': v2,
        'fwhm': fwhm,
    }
  return res

def fwhm(xiH, sigH, m, dt, lent, winlen, period):
  leftxiH, leftpsf, rightxiH, rightpsf = psf_xiH(xiH, sigH, m, dt, lent, winlen, period)
  resl = fwhm_and_psf_peaks(leftxiH, leftpsf)
  resr = fwhm_and_psf_peaks(rightxiH, rightpsf)
  return max(resl['fwhm'], resr['fwhm'])
